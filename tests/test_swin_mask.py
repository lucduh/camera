from types import SimpleNamespace

import torch

from donut_camera.swin_mask import (
    ShiftMaskCache,
    historical_mask,
    install_cached_masks,
    restore_transformers_masks,
    target_device_mask,
)


def block(*, shift_size: int = 5):
    return SimpleNamespace(window_size=10, shift_size=shift_size)


def test_mask_variants_are_exact_on_cpu() -> None:
    shifted = block()
    device = torch.device("cpu")
    historical = historical_mask(shifted, 20, 20, torch.float16, device)
    current = target_device_mask(shifted, 20, 20, torch.float16, device)
    assert historical is not None
    assert torch.equal(historical, current)
    assert historical.shape == (4, 100, 100)


def test_unshifted_block_has_no_mask() -> None:
    unshifted = block(shift_size=0)
    cache = ShiftMaskCache()
    assert (
        historical_mask(unshifted, 20, 20, torch.float32, torch.device("cpu")) is None
    )
    assert cache.get(unshifted, 20, 20, torch.float32, torch.device("cpu")) is None
    assert cache.hits == 0
    assert cache.misses == 0


def test_cache_reuses_masks_and_separates_keys() -> None:
    shifted = block()
    cache = ShiftMaskCache()
    first = cache.get(shifted, 20, 20, torch.float32, torch.device("cpu"))
    second = cache.get(shifted, 20, 20, torch.float32, torch.device("cpu"))
    other_dtype = cache.get(shifted, 20, 20, torch.float16, torch.device("cpu"))
    other_shape = cache.get(shifted, 30, 20, torch.float32, torch.device("cpu"))
    assert first is second
    assert other_dtype is not first
    assert other_shape is not first
    assert cache.hits == 1
    assert cache.misses == 3
    assert len(cache.values) == 3


def test_install_and_restore_cached_method() -> None:
    shifted = block()
    unshifted = block(shift_size=0)
    model = SimpleNamespace(
        encoder=SimpleNamespace(
            encoder=SimpleNamespace(
                layers=[SimpleNamespace(blocks=[shifted, unshifted])]
            )
        )
    )
    cache = ShiftMaskCache()
    install_cached_masks(model, cache)
    first = shifted.get_attn_mask(20, 20, torch.float32, torch.device("cpu"))
    second = shifted.get_attn_mask(20, 20, torch.float32, torch.device("cpu"))
    assert first is second
    assert unshifted.get_attn_mask(20, 20, torch.float32, torch.device("cpu")) is None
    restore_transformers_masks(model)
    assert "get_attn_mask" not in shifted.__dict__
    assert "get_attn_mask" not in unshifted.__dict__
