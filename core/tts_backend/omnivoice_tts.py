import os
import threading
from pathlib import Path

from pydub import AudioSegment
import soundfile as sf

from core.utils import *
from core.utils.models import _AUDIO_REFERS_DIR

DEFAULT_MODEL = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000
# Custom uploaded reference lives outside output/ so Archive to 'history'
# (which moves everything under output/) never deletes it.
CUSTOM_REF_DIR = "custom_voice"
CUSTOM_REF_FILE = "omnivoice_custom_ref.wav"
# Normalized copy of a reference configured by hand (mp3, other rate/channels).
NORMALIZED_REF_FILE = "omnivoice_custom_ref_norm.wav"
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
_custom_ref_key = None
_custom_ref_path = None


def _file_identity(path):
    """What the caches compare to notice a replaced file: path, mtime and size."""
    stat = os.stat(path)
    return (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)


def _resolve_device(device, torch):
    from core.utils.gpu_utils import resolve_torch_device
    # Explicit values ('cuda:1', 'mps', 'xpu', 'cpu') pass through;
    # 'auto'/'cuda' pick the GPU with the most free VRAM.
    resolved = resolve_torch_device(device or "auto", torch)
    # Legacy auto order was cuda:0 -> mps -> cpu; keep mps fallback for
    # torch fakes/tests that lack resolve support.
    if resolved == "cpu":
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return resolved


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
    key = (out_dir, tuple(task_df["number"].tolist()), tuple(str(x) for x in task_df["origin"].tolist()))
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


def custom_reference_path():
    """Persistent path of the uploaded custom reference (outside output/)."""
    return os.path.join(CUSTOM_REF_DIR, CUSTOM_REF_FILE)


def save_custom_reference(uploaded_bytes, filename=""):
    """Save an uploaded clip as 24 kHz mono WAV for custom voice cloning.

    Returns (path, kept_seconds, original_seconds). Trims to
    MAX_REFERENCE_SECONDS so a long upload behaves like the stable reference;
    callers must tell the user when trimming happened, because the transcript
    has to match exactly the kept part.
    """
    os.makedirs(CUSTOM_REF_DIR, exist_ok=True)
    tmp_path = os.path.join(CUSTOM_REF_DIR, f"_upload_tmp{os.path.splitext(filename or '')[1].lower() or '.wav'}")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_bytes)
    try:
        clip = AudioSegment.from_file(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    duration = len(clip) / 1000.0
    if duration < MIN_REFERENCE_SECONDS:
        raise ValueError(
            f"Reference audio is only {duration:.1f}s; please upload at least "
            f"{MIN_REFERENCE_SECONDS:.0f}s of clear speech of one speaker."
        )
    if len(clip) > int(MAX_REFERENCE_SECONDS * 1000):
        clip = clip[: int(MAX_REFERENCE_SECONDS * 1000)]
    out_path = custom_reference_path()
    clip.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2).export(out_path, format="wav")
    return out_path, len(clip) / 1000.0, duration


def _custom_reference():
    """The user-uploaded reference audio + its transcript, for the whole video.

    The model gets a 24 kHz mono WAV even when the config points at an mp3 or
    another rate. The normalized copy is cached per file version, so the file
    is never decoded or rewritten once per line.
    """
    global _custom_ref_key, _custom_ref_path
    ref_audio = str(load_key_or("omnivoice.custom_ref_audio", "") or "")
    ref_text = str(load_key_or("omnivoice.custom_ref_text", "") or "").strip()
    if not ref_audio or not os.path.exists(ref_audio):
        raise FileNotFoundError(
            "Custom reference audio is not set. Select OmniVoice -> Custom voice, "
            "upload a 3-10 s clip and enter its transcript."
        )
    if not ref_text:
        raise ValueError(
            "Custom reference text is empty. Enter the exact transcript of the uploaded audio."
        )
    key = _file_identity(ref_audio)
    if _custom_ref_key != key or _custom_ref_path is None:
        normalized_path = ref_audio
        try:
            clip = AudioSegment.from_file(ref_audio)
            needs_conversion = ref_audio.lower().endswith((".mp3", ".m4a", ".flac", ".ogg"))
            needs_conversion = needs_conversion or (clip.channels, clip.frame_rate, clip.sample_width) != (1, SAMPLE_RATE, 2)
            if needs_conversion:
                os.makedirs(CUSTOM_REF_DIR, exist_ok=True)
                normalized_path = os.path.join(CUSTOM_REF_DIR, NORMALIZED_REF_FILE)
                clip.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2).export(normalized_path, format="wav")
        except Exception as exc:
            rprint(f"[yellow]⚠️ Could not normalize the custom reference ({exc}); using it as uploaded.[/yellow]")
            normalized_path = ref_audio
        _custom_ref_key = key
        _custom_ref_path = normalized_path
    return _custom_ref_path, ref_text


def _voice_clone_kwargs(model, ref_audio, ref_text):
    """Encode the reference once and reuse it; fall back to re-passing the audio."""
    global _voice_prompt, _voice_prompt_key
    key = (*_file_identity(ref_audio), ref_text)
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
    elif refer_mode == "custom":
        ref_audio, ref_text = _custom_reference()
        gen_kwargs = _voice_clone_kwargs(model, ref_audio, ref_text)
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
