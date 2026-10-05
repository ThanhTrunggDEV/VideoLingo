from types import SimpleNamespace

from core.utils import gpu_utils as gpu


def _fake_torch(free_list, available=True, bf16=True, capability=(8, 0)):
    cuda = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: len(free_list),
        mem_get_info=lambda idx: (free_list[idx], 8 * 1024 ** 3),
        get_device_properties=lambda idx: SimpleNamespace(
            total_memory=8 * 1024 ** 3, name=f"GPU{idx}"),
        get_device_capability=lambda idx: capability,
        is_bf16_supported=lambda: bf16,
        set_device=lambda idx: None,
        get_device_name=lambda idx: f"FakeGPU{idx}",
    )
    return SimpleNamespace(
        cuda=cuda, float16="fp16", bfloat16="bf16", float32="fp32",
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))


def test_best_gpu_is_most_free_vram():
    fake = _fake_torch([1 * 1024 ** 3, 6 * 1024 ** 3, 2 * 1024 ** 3])
    assert gpu.get_best_cuda_index(fake) == 1
    assert gpu.resolve_torch_device("auto", fake) == "cuda:1"
    assert gpu.resolve_torch_device("cuda", fake) == "cuda:1"


def test_single_gpu_needs_no_ffmpeg_flag_but_multi_does():
    assert gpu.ffmpeg_gpu_args(_fake_torch([4 * 1024 ** 3])) == []
    assert gpu.ffmpeg_gpu_args(_fake_torch([1 * 1024 ** 3, 6 * 1024 ** 3])) == ["-gpu", "1"]


def test_explicit_device_is_respected():
    fake = _fake_torch([1 * 1024 ** 3, 6 * 1024 ** 3])
    assert gpu.resolve_torch_device("cuda:0", fake) == "cuda:0"
    assert gpu.resolve_torch_device("cpu", fake) == "cpu"


def test_cpu_torch_never_resolves_to_cuda():
    fake = _fake_torch([], available=False)
    assert gpu.get_best_cuda_index(fake) is None
    assert gpu.resolve_torch_device("auto", fake) == "cpu"


def test_old_fake_without_device_count_falls_back_to_cuda_0():
    # Compatibility with existing test fakes (only is_available/is_bf16_supported).
    fake = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
        bfloat16="bfloat16", float16="float16", float32="float32")
    assert gpu.get_best_cuda_index(fake) == 0
    assert gpu.resolve_torch_device("auto", fake) == "cuda:0"


def test_pre_pascal_gpu_gets_fp16():
    fake = _fake_torch([4 * 1024 ** 3], bf16=True, capability=(7, 5))
    assert gpu.torch_dtype_for_device("cuda:0", fake) == "fp16"
    fake_amp = _fake_torch([4 * 1024 ** 3], bf16=True, capability=(8, 0))
    assert gpu.torch_dtype_for_device("cuda:0", fake_amp) == "bf16"
