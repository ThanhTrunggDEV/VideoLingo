"""Custom reference voices: cache invalidation when the clip changes, one-shot normalization."""
import os
import time

import pandas as pd
from pydub.generators import Sine


def _write_clip(path, duration_ms=4000, rate=16000, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    segment = Sine(300, sample_rate=rate).to_audio_segment(duration=duration_ms)
    if channels == 2:
        segment = segment.set_channels(2)
    segment.export(str(path), format="wav")
    return str(path)


class FakeVieneuModel:
    def __init__(self):
        self.enrollments = []

    def add_voice(self, name, ref_audio, denoise=True):
        self.enrollments.append((name, ref_audio, denoise))
        return name


class FakeOmniVoiceModel:
    def __init__(self):
        self.prompts = []

    def create_voice_clone_prompt(self, ref_audio, ref_text):
        self.prompts.append((ref_audio, ref_text))
        return {"ref_audio": ref_audio}


# ---------------------------------------------------------------------------
# VieNeu-TTS: the enrollment follows the reference file
# ---------------------------------------------------------------------------

def test_vieneu_reenrolls_when_the_reference_file_changes(tmp_path, monkeypatch):
    from core.tts_backend import vieneu_tts as vt

    monkeypatch.setattr(vt, "_enrolled_key", None)
    ref = _write_clip(tmp_path / "vieneu_custom_ref.wav")
    model = FakeVieneuModel()

    assert vt._enrolled_voice(model, ref, True) == vt.ENROLLED_NAME
    assert len(model.enrollments) == 1

    vt._enrolled_voice(model, ref, True)
    assert len(model.enrollments) == 1  # unchanged file: no new enrollment

    time.sleep(0.01)
    _write_clip(tmp_path / "vieneu_custom_ref.wav", duration_ms=3500)
    vt._enrolled_voice(model, ref, True)
    assert len(model.enrollments) == 2  # replaced file: enrolled again

    vt._enrolled_voice(model, ref, False)
    assert len(model.enrollments) == 3  # denoise toggled: enrolled again


def test_vieneu_stable_reference_is_built_once_per_video(tmp_path, monkeypatch):
    from core.tts_backend import vieneu_tts as vt

    monkeypatch.setattr(vt, "_stable_ref_key", None)
    monkeypatch.setattr(vt, "_stable_ref_value", None)
    refers = tmp_path / "refers"
    _write_clip(refers / "1.wav", duration_ms=2000)
    _write_clip(refers / "2.wav", duration_ms=2000)
    monkeypatch.setattr(vt, "_AUDIO_REFERS_DIR", str(refers))
    tasks = pd.DataFrame({"number": [1, 2], "origin": ["first line", "second line"]})
    out_dir = tmp_path / "tmp"
    out_dir.mkdir()

    first = vt._stable_reference(tasks, str(out_dir))
    mtime = os.path.getmtime(first)
    time.sleep(0.01)

    assert vt._stable_reference(tasks, str(out_dir)) == first
    assert os.path.getmtime(first) == mtime  # not rebuilt per line


# ---------------------------------------------------------------------------
# OmniVoice: the clone prompt follows the reference file
# ---------------------------------------------------------------------------

def test_omnivoice_reencodes_the_prompt_when_the_reference_changes(tmp_path, monkeypatch):
    from core.tts_backend import omnivoice_tts as ot

    monkeypatch.setattr(ot, "_voice_prompt", None)
    monkeypatch.setattr(ot, "_voice_prompt_key", None)
    ref = _write_clip(tmp_path / "omnivoice_custom_ref.wav", rate=24000)
    model = FakeOmniVoiceModel()

    ot._voice_clone_kwargs(model, ref, "hello there")
    ot._voice_clone_kwargs(model, ref, "hello there")
    assert len(model.prompts) == 1  # unchanged file: prompt cache hit

    time.sleep(0.01)
    _write_clip(tmp_path / "omnivoice_custom_ref.wav", duration_ms=3500, rate=24000)
    ot._voice_clone_kwargs(model, ref, "hello there")
    assert len(model.prompts) == 2  # replaced file: re-encoded


def test_omnivoice_custom_reference_is_normalized_once_without_touching_the_source(tmp_path, monkeypatch):
    import soundfile as sf
    from core.tts_backend import omnivoice_tts as ot

    monkeypatch.setattr(ot, "_custom_ref_key", None)
    monkeypatch.setattr(ot, "_custom_ref_path", None)
    monkeypatch.setattr(ot, "CUSTOM_REF_DIR", str(tmp_path / "custom_voice"))
    monkeypatch.setattr(ot, "NORMALIZED_REF_FILE", "omnivoice_custom_ref_norm.wav")
    ref = _write_clip(tmp_path / "manual_ref.wav", rate=44100, channels=2)
    config = {"omnivoice.custom_ref_audio": ref, "omnivoice.custom_ref_text": "hello there"}
    monkeypatch.setattr(ot, "load_key_or", lambda key, default=None: config.get(key, default))
    source_mtime = os.path.getmtime(ref)

    first_path, first_text = ot._custom_reference()
    normalized_mtime = os.path.getmtime(first_path)
    time.sleep(0.01)
    second_path, second_text = ot._custom_reference()

    assert (first_path, first_text) == (second_path, second_text)
    assert os.path.abspath(first_path) != os.path.abspath(ref)
    assert os.path.getmtime(ref) == source_mtime  # user file kept as is
    assert os.path.getmtime(first_path) == normalized_mtime  # normalized once
    info = sf.info(first_path)
    assert (info.samplerate, info.channels) == (ot.SAMPLE_RATE, 1)
