"""Local on-device Vietnamese TTS via the VieNeu-TTS Python SDK (docs.vieneu.io).

Modes (same options as OmniVoice, plus built-in presets):
- preset:  a built-in voice by name (e.g. 'Minh Quân Pro'), no reference needed.
- stable:  first 3-8 s of the video as one reference for all lines.
- dynamic: each line's own clip as its reference.
- custom:  a user-uploaded clip as the reference for the whole video.

Unlike OmniVoice, v3 Turbo needs no reference transcript: cloning takes only
``ref_audio`` (3-8 s, auto-denoised and trimmed to <= 8 s by the SDK).
The model is heavy: it is loaded once and reused for every line; the
stable/custom reference is enrolled once via ``add_voice`` (in-memory only,
never ``save_voices``) and generation is serialized with a lock.
"""

import inspect
import os
import shutil
import threading
from pathlib import Path

from pydub import AudioSegment

from core.utils import *
from core.utils.models import _AUDIO_REFERS_DIR

# v3 Turbo default voice (docs "Editors' picks"); "Minh Quân" stays an alias.
DEFAULT_VOICE = "Minh Quân Pro"
# Custom uploaded reference lives outside output/ so Archive to 'history'
# (which moves everything under output/) never deletes it.
CUSTOM_REF_DIR = "custom_voice"
CUSTOM_REF_FILE = "vieneu_custom_ref.wav"
# In-memory enrolled name for the stable/custom reference (never persisted
# with save_voices, so the user's voices file is left untouched).
ENROLLED_NAME = "VideoLingoRef"
# VieNeu-TTS clones from a 3-8 s clip of one speaker (SDK trims to <= 8 s).
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 8.0

_model = None
_model_lock = threading.Lock()
_enrolled_key = None
_stable_ref_key = None
_stable_ref_value = None


def _file_identity(path):
    """What the caches compare to notice a replaced file: path, mtime and size."""
    stat = os.stat(path)
    return (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)


def _require_espeak():
    """VieNeu-TTS phonemizes with eSpeak NG, a system package pip cannot provide."""
    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        raise RuntimeError(
            "VieNeu-TTS needs the eSpeak NG system package (Ubuntu/Debian: "
            "`sudo apt install espeak-ng`, macOS: `brew install espeak`, "
            "Windows: .msi from github.com/espeak-ng/espeak-ng/releases)."
        )


def _get_model(settings):
    """Load VieNeu once and reuse it; supports current and older SDK signatures."""
    global _model
    if _model is not None:
        return _model
    _require_espeak()
    try:
        from vieneu import Vieneu
    except ImportError as exc:
        raise ImportError(
            "VieNeu-TTS is not installed. Run `pip install vieneu` "
            "(GPU: `pip install \"vieneu[cuda]\"` after a CUDA torch). "
            "See https://docs.vieneu.io/docs/getting-started/installation."
        ) from exc
    wanted = {}
    mode = str(settings.get("mode") or "standard")
    if mode != "standard":
        wanted["mode"] = mode
    backend = str(settings.get("backend") or "auto").lower()
    if backend not in ("auto", ""):
        wanted["backend"] = backend
    precision = str(settings.get("precision") or "fp32").lower()
    if precision in ("fp32", "int8"):
        wanted["precision"] = precision
    try:
        params = inspect.signature(Vieneu).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            wanted = {k: v for k, v in wanted.items() if k in params}
    except (TypeError, ValueError):
        pass
    rprint(f"[yellow]Loading VieNeu-TTS (mode={mode}, backend={backend}, "
           f"precision={precision})...[/yellow]")
    _model = Vieneu(**wanted)
    return _model


def _stable_reference(task_df, out_dir):
    """The first 3-8 s of the video as one reference for all lines: WAV path."""
    global _stable_ref_key, _stable_ref_value
    key = (out_dir, tuple(task_df["number"].tolist()), tuple(str(x) for x in task_df["origin"].tolist()))
    if _stable_ref_key == key and _stable_ref_value is not None:
        return _stable_ref_value
    audio = AudioSegment.empty()
    for _, row in task_df.iterrows():
        path = os.path.join(_AUDIO_REFERS_DIR, f"{row['number']}.wav")
        if not os.path.exists(path):
            continue
        clip = AudioSegment.from_file(path)
        if len(audio) + len(clip) > int(MAX_REFERENCE_SECONDS * 1000):
            if len(audio) == 0:  # one line that is longer than the limit
                audio = clip[: int(MAX_REFERENCE_SECONDS * 1000)]
            break
        audio += clip
        if len(audio) >= int(MIN_REFERENCE_SECONDS * 1000):
            break
    if len(audio) < int(MIN_REFERENCE_SECONDS * 1000):
        raise FileNotFoundError(
            f"No usable reference audio of the video in {_AUDIO_REFERS_DIR} "
            f"(need >= {MIN_REFERENCE_SECONDS:.0f}s of speech)."
        )
    ref_path = os.path.join(out_dir, "vieneu_reference.wav")
    audio.set_channels(1).set_frame_rate(16000).set_sample_width(2).export(ref_path, format="wav")
    _stable_ref_key = key
    _stable_ref_value = ref_path
    return ref_path


def _dynamic_reference(task_df, number):
    """The reference clip of the current line, falling back to the first clip."""
    path = os.path.join(_AUDIO_REFERS_DIR, f"{number}.wav")
    if not os.path.exists(path):
        path = os.path.join(_AUDIO_REFERS_DIR, "1.wav")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference audio not found in {_AUDIO_REFERS_DIR}")
    return path


def custom_reference_path():
    """Persistent path of the uploaded custom reference (outside output/)."""
    return os.path.join(CUSTOM_REF_DIR, CUSTOM_REF_FILE)


def save_custom_reference(uploaded_bytes, filename=""):
    """Save an uploaded clip as mono WAV for custom voice cloning.

    Returns (path, kept_seconds, original_seconds). Trims to
    MAX_REFERENCE_SECONDS; v3 needs no transcript, so only the audio is stored.
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
    clip.set_channels(1).set_frame_rate(16000).set_sample_width(2).export(out_path, format="wav")
    return out_path, len(clip) / 1000.0, duration


def _custom_reference():
    """The user-uploaded reference clip for the whole video (no transcript needed)."""
    ref_audio = str(load_key_or("vieneu.custom_ref_audio", "") or "")
    if not ref_audio or not os.path.exists(ref_audio):
        raise FileNotFoundError(
            "Custom reference audio is not set. Select VieNeu-TTS -> Custom voice "
            "and upload a 3-8 s clip."
        )
    return ref_audio


def _enrolled_voice(model, ref_audio, denoise):
    """Enroll the stable/custom reference once per file version, reuse it by name afterwards."""
    global _enrolled_key
    key = (*_file_identity(ref_audio), bool(denoise))
    if _enrolled_key != key:
        try:
            model.add_voice(ENROLLED_NAME, ref_audio, denoise=denoise)
        except TypeError:
            # Older SDK without the denoise kwarg.
            model.add_voice(ENROLLED_NAME, ref_audio)
        _enrolled_key = key
    return ENROLLED_NAME


def _infer_kwargs(model, refer_mode, task_df, number, out_dir, settings):
    """infer() keyword arguments for one line."""
    denoise = bool(settings.get("denoise", True))
    if refer_mode == "preset":
        voice = str(settings.get("voice") or DEFAULT_VOICE).strip() or DEFAULT_VOICE
        return {"voice": voice}
    if refer_mode == "dynamic":
        return {"ref_audio": _dynamic_reference(task_df, number), "denoise": denoise}
    if refer_mode == "custom":
        return {"voice": _enrolled_voice(model, _custom_reference(), denoise)}
    # stable (default): first 3-8 s of the video, enrolled once.
    return {"voice": _enrolled_voice(model, _stable_reference(task_df, out_dir), denoise)}


def _infer(model, text, **kwargs):
    """infer() with denoise fallback for older SDKs without that kwarg."""
    try:
        return model.infer(text, **kwargs)
    except TypeError as exc:
        if "denoise" in kwargs and "denoise" not in str(exc):
            raise
        if "denoise" not in kwargs:
            raise
        kwargs.pop("denoise")
        return model.infer(text, **kwargs)


@except_handler("Failed to generate audio using VieNeu-TTS", retry=3, delay=1)
def vieneu_tts(text, save_as, number=None, task_df=None):
    try:
        settings = load_key("vieneu")
    except KeyError as exc:
        raise KeyError(
            "Missing 'vieneu' section in config.yaml. Update config.yaml from the "
            "template or select VieNeu-TTS in the sidebar once."
        ) from exc
    model = _get_model(settings)
    refer_mode = str(settings.get("refer_mode") or "stable").lower()
    if refer_mode not in ("preset", "stable", "dynamic", "custom"):
        raise ValueError(
            f"vieneu.refer_mode must be 'preset', 'stable', 'dynamic' or 'custom', "
            f"got {refer_mode!r}"
        )
    out_dir = str(Path(save_as).parent)
    gen_kwargs = _infer_kwargs(model, refer_mode, task_df, number, out_dir, settings)

    # TTS may run in a ThreadPoolExecutor; serialize model access.
    with _model_lock:
        audio = _infer(model, text, **gen_kwargs)
        Path(save_as).parent.mkdir(parents=True, exist_ok=True)
        model.save(audio, save_as)
    print(f"Audio saved to {save_as}")


if __name__ == "__main__":
    import pandas as pd
    from core.utils.models import _8_1_AUDIO_TASK

    vieneu_tts("Xin chào, đây là VieNeu-TTS!", "test.wav", 1, pd.read_excel(_8_1_AUDIO_TASK))
