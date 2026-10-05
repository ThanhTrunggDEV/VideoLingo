"""Central GPU auto-selection: pick the CUDA device with the most free VRAM.

Before this module every backend hardcoded ``cuda:0``:
- ``core/asr_backend/qwen_asr_local.py::_torch_device``
- ``core/tts_backend/omnivoice_tts.py::_resolve_device``
- ``core/asr_backend/whisperX_local.py`` (``get_device_properties(0)``)
- ``core/asr_backend/demucs_vl.py`` (``device="cuda"`` -> torch current device 0)

``resolve_torch_device("auto")`` now returns ``cuda:<best>`` instead of
always ``cuda:0``. Explicit values (``cuda:1``, ``cpu``, ``mps``, ``xpu``)
are still respected. ``"cuda"`` without an index also resolves to the best GPU.
"""
from __future__ import annotations

import os
import re
import subprocess


def parse_cuda_index(device: str | None) -> int | None:
    """Return the CUDA index from 'cuda', 'cuda:N', or torch.device, else None."""
    if device is None:
        return None
    text = str(device).strip().lower()
    if text == "cuda":
        return -1  # bare "cuda": caller should resolve to best
    match = re.fullmatch(r"cuda:(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _best_index_via_torch(torch_mod=None) -> int | None:
    """Pick the CUDA index with the most free memory via torch, or None."""
    try:
        import torch as torch_real
    except Exception:
        torch_real = None
    torch_mod = torch_mod if torch_mod is not None else torch_real
    if torch_mod is None:
        return None
    try:
        cuda = getattr(torch_mod, "cuda", None)
        if cuda is None or not cuda.is_available():
            return None
        count_fn = getattr(cuda, "device_count", None)
        try:
            count = int(count_fn()) if callable(count_fn) else 1
        except Exception:
            count = 1
        if count <= 0:
            return None
        if count == 1:
            return 0
        # Multi-GPU: compare free memory per device.
        best_idx, best_free = 0, -1
        for idx in range(count):
            try:
                free, _total = cuda.mem_get_info(idx)  # torch >= 1.13
            except Exception:
                try:
                    props = cuda.get_device_properties(idx)
                    free = int(getattr(props, "total_memory", 0))
                except Exception:
                    free = 0
            if free is None:
                free = 0
            if int(free) > int(best_free):
                best_free, best_idx = int(free), idx
        return best_idx
    except Exception:
        return None


def _best_index_via_smi() -> int | None:
    """Fallback when torch has no CUDA: parse nvidia-smi free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    best_idx, best_free = None, -1
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx, free = int(parts[0]), int(float(parts[1]))
        except ValueError:
            continue
        if free > best_free:
            best_free, best_idx = free, idx
    return best_idx


def get_best_cuda_index(torch_mod=None) -> int | None:
    """Best CUDA device index torch can actually use, else None.

    Strict: when torch is importable/present but ``cuda.is_available()`` is
    False (CPU-only torch), returns None even if ``nvidia-smi`` sees a GPU --
    torch could not run on it anyway. System-wide (ffmpeg/installer) callers
    that do not need torch should use :func:`get_system_best_cuda_index`.
    Honors CUDA_VISIBLE_DEVICES implicitly (torch only sees visible devices).
    """
    return _best_index_via_torch(torch_mod)


def get_system_best_cuda_index(torch_mod=None) -> int | None:
    """Best CUDA index on the machine, via torch then nvidia-smi fallback.

    For non-torch consumers (ffmpeg NVENC ``-gpu`` flag, installer/launch
    messages) where a GPU is useful even when the installed torch is
    CPU-only.
    """
    idx = _best_index_via_torch(torch_mod)
    if idx is not None:
        return idx
    # Only fall back to nvidia-smi when torch itself cannot tell us --
    # i.e. torch missing, or torch present but CUDA unusable. Never override
    # an explicit torch "no CUDA" with a torch-unusable index for torch
    # callers (they use get_best_cuda_index); this is for system callers.
    try:
        import torch as torch_real
    except Exception:
        torch_real = None
    mod = torch_mod if torch_mod is not None else torch_real
    if mod is not None:
        try:
            cuda = getattr(mod, "cuda", None)
            if cuda is not None and callable(getattr(cuda, "is_available", None)):
                if not bool(cuda.is_available()):
                    # CPU-only torch: smi fallback is still meaningful for
                    # ffmpeg, so allow it here (system scope only).
                    pass
        except Exception:
            pass
    return _best_index_via_smi()


def get_best_cuda_device(torch_mod=None) -> str | None:
    """'cuda:N' for the GPU with the most free VRAM, else None."""
    idx = get_best_cuda_index(torch_mod)
    return f"cuda:{idx}" if idx is not None and idx >= 0 else None


def use_best_cuda_device(torch_mod=None) -> int | None:
    """Point torch's *current* device at the best GPU so bare 'cuda' uses it.

    Returns the index selected, or None. Never raises.
    """
    idx = get_best_cuda_index(torch_mod)
    if idx is None:
        return None
    try:
        import torch as torch_real
    except Exception:
        torch_real = None
    mod = torch_mod if torch_mod is not None else torch_real
    try:
        if mod is not None and hasattr(mod.cuda, "set_device"):
            mod.cuda.set_device(idx)
    except Exception:
        pass
    return idx


def resolve_torch_device(preference: str | None = "auto", torch_mod=None) -> str:
    """Resolve 'auto'/explicit device to a concrete torch device string.

    - explicit 'cuda:N', 'mps', 'xpu', 'cpu' (and 'cuda:N' with spaces/case) pass through
    - 'cuda' (no index) and 'auto' resolve to the best CUDA GPU when available
    - 'auto' falls back to mps -> xpu -> cpu
    - CUDA_VISIBLE_DEVICES with a single visible GPU still yields 'cuda:0'
      (torch-side index), which is correct for that process
    """
    text = (str(preference).strip().lower() if preference else "auto") or "auto"
    if text not in ("auto", "cuda"):
        return text

    best = get_best_cuda_device(torch_mod)
    if best is not None:
        return best

    # No CUDA: keep the old auto fallback order. torch_mod may be a test fake.
    try:
        import torch as torch_real
    except Exception:
        torch_real = None
    mod = torch_mod if torch_mod is not None else torch_real
    try:
        if mod is not None:
            backends = getattr(mod, "backends", None)
            mps = getattr(backends, "mps", None) if backends is not None else None
            if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
                return "mps"
    except Exception:
        pass
    # Intel XPU (torch>=2.x): torch.xpu.is_available()
    try:
        if mod is not None:
            xpu = getattr(mod, "xpu", None)
            if xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available():
                return "xpu"
    except Exception:
        pass
    return "cpu"


def torch_dtype_for_device(device: str, torch_mod=None):
    """float32 on CPU, otherwise bf16 when supported else float16."""
    try:
        import torch as torch_real
    except Exception:
        torch_real = None
    mod = torch_mod if torch_mod is not None else torch_real
    if mod is None:
        return None
    if str(device).strip().lower() == "cpu":
        return mod.float32
    try:
        # Ampere+ (capability >= 8.0) generally supports bf16; is_bf16_supported
        # checks the *current* device, so capability is the per-GPU-safe check.
        idx = parse_cuda_index(device)
        if idx is not None and idx >= 0:
            try:
                major, _minor = mod.cuda.get_device_capability(idx)
                if int(major) < 8:
                    return mod.float16
            except Exception:
                pass
        if bool(mod.cuda.is_bf16_supported()):
            return mod.bfloat16
    except Exception:
        pass
    try:
        return mod.float16
    except Exception:
        return None


def cuda_device_name(index: int, torch_mod=None) -> str | None:
    """Human-readable GPU name for logs, or None."""
    try:
        import torch as torch_real
    except Exception:
        torch_real = None
    mod = torch_mod if torch_mod is not None else torch_real
    if mod is None:
        return None
    try:
        return mod.cuda.get_device_name(index)
    except Exception:
        return None


def cuda_supports_flash(device: str, torch_mod=None) -> bool | None:
    """Whether a CUDA device supports FlashAttention (Ampere+, capability >= 8.0).

    Returns True/False when the capability is known, None when it cannot be
    determined (torch missing, non-CUDA device, or query failed). Callers
    should only change behavior on an explicit False and keep the default
    fast path otherwise.
    """
    idx = parse_cuda_index(device)
    if idx is None or idx < 0:
        return None
    try:
        import torch as torch_real
    except Exception:
        torch_real = None
    mod = torch_mod if torch_mod is not None else torch_real
    if mod is None:
        return None
    try:
        major, _minor = mod.cuda.get_device_capability(idx)
        return int(major) >= 8
    except Exception:
        return None


def attn_implementation_for_device(device: str, torch_mod=None) -> str | None:
    """'eager' for pre-Ampere CUDA GPUs (no FlashAttention), else None.

    transformers' SDPA path dispatches to the flash kernel, which raises
    ``RuntimeError: FlashAttention only supports Ampere GPUs or newer`` on
    older cards (Pascal/Turing/...). Loading those models with
    ``attn_implementation="eager"`` avoids the flash path entirely and runs
    correctly everywhere, at some speed cost. Ampere+ keeps the default
    (fast) implementation, so pass the result only when it is not None.
    """
    try:
        if str(device).strip().lower() in ("cpu", "mps", "xpu"):
            return None
    except Exception:
        return None
    supported = cuda_supports_flash(device, torch_mod)
    return "eager" if supported is False else None


def describe_torch_device(device: str, torch_mod=None) -> str:
    """'cuda:1 (RTX A4000, cap 8.6)' style description for logs."""
    idx = parse_cuda_index(device)
    if idx is None:
        return str(device)
    name = cuda_device_name(idx, torch_mod) or "CUDA GPU"
    cap_note = ""
    try:
        import torch as torch_real
    except Exception:
        torch_real = None
    mod = torch_mod if torch_mod is not None else torch_real
    try:
        if mod is not None:
            major, minor = mod.cuda.get_device_capability(idx)
            cap_note = f", cap {major}.{minor}"
    except Exception:
        pass
    return f"{device} ({name}{cap_note})"


def ffmpeg_gpu_args(torch_mod=None) -> list[str]:
    """Extra ffmpeg args selecting the best GPU for h264_nvenc, else [].

    Single-GPU machines need no '-gpu' flag (defaults to 0). Multi-GPU
    machines get ['-gpu', '<best>'] so NVENC runs where torch runs.
    Uses the system-wide best GPU (works even with CPU-only torch).
    """
    idx = get_system_best_cuda_index(torch_mod)
    if idx is None or idx <= 0:
        return []
    return ["-gpu", str(idx)]


def describe_best_gpu(torch_mod=None) -> str:
    """One-line description for logs, e.g. 'cuda:0 (GTX 1650, 3532 MiB free)'."""
    idx = get_system_best_cuda_index(torch_mod)
    if idx is None:
        return "cpu (no CUDA GPU detected)"
    name = cuda_device_name(idx, torch_mod) or "CUDA GPU"
    free_note = ""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={idx}", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            free_note = f", {result.stdout.strip().splitlines()[0].strip()} MiB free"
    except Exception:
        pass
    # torch-side free memory is more accurate when nvidia-smi is missing.
    if not free_note:
        try:
            import torch as torch_real
            mod = torch_mod if torch_mod is not None else torch_real
            if mod is not None:
                free, total = mod.cuda.mem_get_info(idx)
                free_note = f", {int(free) // (1024 * 1024)} MiB free"
        except Exception:
            pass
    # Respect single-GPU visibility remapping: torch index may differ from
    # physical nvidia-smi id, but the name still identifies the card.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        free_note += f" [CUDA_VISIBLE_DEVICES={visible}]"
    return f"cuda:{idx} ({name}{free_note})"
