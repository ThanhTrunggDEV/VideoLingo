import os
from importlib.util import find_spec
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from core.utils import *
from core.utils.models import *
import pandas as pd
import soundfile as sf
console = Console()
from core.utils.models import *

def time_to_samples(time_str, sr):
    """Unified time conversion function"""
    h, m, s = time_str.split(':')
    s, ms = s.split(',') if ',' in s else (s, '0')
    seconds = int(h) * 3600 + int(m) * 60 + float(s) + float(ms) / 1000
    return int(seconds * sr)

def extract_audio(audio_data, sr, start_time, end_time, out_file):
    """Simplified audio extraction function"""
    start = time_to_samples(start_time, sr)
    end = time_to_samples(end_time, sr)
    sf.write(out_file, audio_data[start:end], sr)

def references_needed() -> bool:
    """Whether the configured dubbing method reads the clips of output/audio/refers."""
    method = str(load_key("tts_method"))
    if method == "omnivoice":
        return str(load_key_or("omnivoice.refer_mode", "stable") or "stable") != "custom"
    if method == "vieneu":
        mode = str(load_key_or("vieneu.refer_mode", "stable") or "stable")
        return mode not in ("preset", "custom")
    return True

def extract_refer_audio_main():
    # Respect the dubbing settings: demucs still produces the background track
    # of the final video, but disabled means raw audio and no background.
    if bool(load_key_or("demucs", False)) and find_spec("demucs") is not None:
        from core.asr_backend.demucs_vl import demucs_audio
        demucs_audio()  # Dubbing normally keeps the original background track.
        reference_audio = _VOCAL_AUDIO_FILE
    else:
        if find_spec("demucs") is None:
            message = ("Vocal separation is unavailable; using the original audio for references. "
                       "The dubbed video will not keep the original background sound.")
        else:
            message = ("Vocal separation is off in the config; using the original audio. "
                       "The dubbed video will not keep the original background sound.")
        rprint(Panel(message, title="Info", border_style="yellow"))
        reference_audio = _RAW_AUDIO_FILE
    if os.path.exists(os.path.join(_AUDIO_REFERS_DIR, '1.wav')):
        rprint(Panel("Reference audio already exists, skipping extraction", title="Info", border_style="blue"))
        return
    if not references_needed():
        rprint(Panel("The selected dubbing mode does not use the reference clips of the video, "
                     "skipping extraction", title="Info", border_style="blue"))
        return

    # Create output directory
    os.makedirs(_AUDIO_REFERS_DIR, exist_ok=True)
    
    # Read task file and audio data
    df = pd.read_excel(_8_1_AUDIO_TASK)
    data, sr = sf.read(reference_audio)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Extracting audio segments...", total=len(df))
        
        for _, row in df.iterrows():
            out_file = os.path.join(_AUDIO_REFERS_DIR, f"{row['number']}.wav")
            extract_audio(data, sr, row['start_time'], row['end_time'], out_file)
            progress.update(task, advance=1)
            
    rprint(Panel(f"Audio segments saved to {_AUDIO_REFERS_DIR}", title="Success", border_style="green"))

if __name__ == "__main__":
    extract_refer_audio_main()
