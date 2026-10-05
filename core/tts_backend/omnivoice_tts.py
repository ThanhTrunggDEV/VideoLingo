import os
import threading
from pathlib import Path

from pydub import AudioSegment
import soundfile as sf

from core.utils import *
from core.utils.models import _AUDIO_REFERS_DIR

DEFAULT_MODEL = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000
# OmniVoice recommends 3-10 s of clear speech of one speaker as the reference
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 10.0

# The model is heavy: load it once and reuse it for every line, and keep the
# reference voice prompt cached for the whole video (encoding it per line would
# slow long videos down considerably).
_model = None
_model_lock = threading.Lock()
_voice_prompt = None
_voice_prompt_key = None
_stable_ref_key = None
_stable_ref_value = None


def _resolve_device(device, torch):
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _torch_dtype(name, torch, device):
    if device == "cpu" or name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _get_model(settings):
    global _model
    if _model is not None:
        return _model
    import torch
    try:
        from omnivoice import OmniVoice
    except ImportError as exc:
        raise ImportError(
            "OmniVoice is not installed. Run `uv run start.py` (Windows/Linux auto-installs "
            "omnivoice) or `python installer.py`. See github.com/k2-fsa/OmniVoice."
        ) from exc

    device = _resolve_device(settings.get("device") or "auto", torch)
    dtype = _torch_dtype(settings.get("dtype") or "float16", torch, device)
    rprint(f"[yellow]Loading OmniVoice on {device} ({dtype})...[/yellow]")
    _model = OmniVoice.from_pretrained(
        settings.get("model") or DEFAULT_MODEL,
        device_map=device,
        dtype=dtype,
    )
    return _model


def _stable_reference(task_df, out_dir):
    """The first 3-10 s of the video as one reference for all lines: (WAV path, text)."""
    global _stable_ref_key, _stable_ref_value
    key = (tuple(task_df["number"].tolist()), tuple(str(x) for x in task_df["origin"].tolist()))
    if _stable_ref_key == key and _stable_ref_value is not None:
        return _stable_ref_value
    audio = AudioSegment.empty()
    texts = []
    for _, row in task_df.iterrows():
        path = os.path.join(_AUDIO_REFERS_DIR, f"{row['number']}.wav")
        if not os.path.exists(path):
            continue
        clip = AudioSegment.from_file(path)
        if len(audio) + len(clip) > MAX_REFERENCE_SECONDS * 1000:
            if not texts:  # one line that is longer than the limit
                audio, texts = clip[: int(MAX_REFERENCE_SECONDS * 1000)], [str(row["origin"])]
            break
        audio += clip
        texts.append(str(row["origin"]))
        if len(audio) >= MIN_REFERENCE_SECONDS * 1000:
            break
    if not texts:
        raise FileNotFoundError(f"No reference audio of the video in {_AUDIO_REFERS_DIR}")
    ref_path = os.path.join(out_dir, "omnivoice_reference.wav")
    audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2).export(ref_path, format="wav")
    _stable_ref_key = key
    _stable_ref_value = (ref_path, " ".join(texts))
    return _stable_ref_value


def _dynamic_reference(task_df, number):
    """The reference clip of the current line, falling back to the first clip."""
    path = os.path.join(_AUDIO_REFERS_DIR, f"{number}.wav")
    if not os.path.exists(path):
        path = os.path.join(_AUDIO_REFERS_DIR, "1.wav")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference audio not found in {_AUDIO_REFERS_DIR}")
    text = task_df.loc[task_df["number"] == number, "origin"].values[0]
    return path, str(text)


def _voice_clone_kwargs(model, ref_audio, ref_text):
    """Encode the reference once and reuse it; fall back to re-passing the audio."""
    global _voice_prompt, _voice_prompt_key
    key = (os.path.abspath(ref_audio), ref_text)
    if _voice_prompt is None or _voice_prompt_key != key:
        try:
            _voice_prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
            _voice_prompt_key = key
        except Exception:
            _voice_prompt = None
            _voice_prompt_key = None
            return {"ref_audio": ref_audio, "ref_text": ref_text}
    return {"voice_clone_prompt": _voice_prompt}


@except_handler("Failed to generate audio using OmniVoice", retry=3, delay=1)
def omnivoice_tts(text, save_as, number=None, task_df=None):
    settings = load_key("omnivoice")
    model = _get_model(settings)
    refer_mode = settings.get("refer_mode") or "stable"
    num_step = int(settings.get("num_step") or 32)
    out_dir = str(Path(save_as).parent)

    if refer_mode == "dynamic":
        ref_audio, ref_text = _dynamic_reference(task_df, number)
        gen_kwargs = {"ref_audio": ref_audio, "ref_text": ref_text}
    else:
        ref_audio, ref_text = _stable_reference(task_df, out_dir)
        gen_kwargs = _voice_clone_kwargs(model, ref_audio, ref_text)

    # TTS may run in a ThreadPoolExecutor; serialize model access.
    with _model_lock:
        audio = model.generate(text=text, num_step=num_step, **gen_kwargs)

    Path(save_as).parent.mkdir(parents=True, exist_ok=True)
    # audio is a list of np.ndarray at 24 kHz
    sf.write(save_as, audio[0], SAMPLE_RATE)
    print(f"Audio saved to {save_as}")


if __name__ == "__main__":
    import pandas as pd
    from core.utils.models import _8_1_AUDIO_TASK

    omnivoice_tts("Hello, welcome to VideoLingo!", "test.wav", 1, pd.read_excel(_8_1_AUDIO_TASK))
