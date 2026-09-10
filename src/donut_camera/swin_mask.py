"""Shifted-window mask variants used by the Chapter 4 reproduction."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import MethodType

import torch

MaskMethod = Callable[
    [object, int, int, torch.dtype, torch.device], torch.Tensor | None
]


def padded_window_shape(height: int, width: int, window_size: int) -> tuple[int, int]:
    """Return the grid size passed to shifted-window mask construction."""
    padded_height = ((height + window_size - 1) // window_size) * window_size
    padded_width = ((width + window_size - 1) // window_size) * window_size
    return padded_height, padded_width


def _partition(mask: torch.Tensor, window_size: int) -> torch.Tensor:
    _, height, width, channels = mask.shape
    windows = mask.view(
        1,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return (
        windows.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size * window_size)
    )


def _fill_mask(
    *,
    height: int,
    width: int,
    window_size: int,
    shift_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    image_mask = torch.zeros((1, height, width, 1), dtype=dtype, device=device)
    height_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    width_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    count = 0
    for height_slice in height_slices:
        for width_slice in width_slices:
            image_mask[:, height_slice, width_slice, :] = count
            count += 1
    windows = _partition(image_mask, window_size)
    mask = windows.unsqueeze(1) - windows.unsqueeze(2)
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


def historical_mask(
    block, height: int, width: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    """Reproduce Transformers 4.37.2: construct on CPU, then transfer."""
    if block.shift_size == 0:
        return None
    mask = _fill_mask(
        height=height,
        width=width,
        window_size=block.window_size,
        shift_size=block.shift_size,
        dtype=dtype,
        device="cpu",
    )
    return mask.to(device)


def target_device_mask(
    block, height: int, width: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    """Reproduce current Transformers construction on the target device."""
    if block.shift_size == 0:
        return None
    return _fill_mask(
        height=height,
        width=width,
        window_size=block.window_size,
        shift_size=block.shift_size,
        dtype=dtype,
        device=device,
    )


@dataclass
class ShiftMaskCache:
    """Per-model cache shared by all shifted blocks."""

    values: dict[tuple, torch.Tensor] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def key(self, block, height, width, dtype, device) -> tuple:
        resolved_device = torch.device(device)
        return (
            height,
            width,
            block.window_size,
            block.shift_size,
            dtype,
            resolved_device.type,
            resolved_device.index,
        )

    def get(
        self, block, height: int, width: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor | None:
        if block.shift_size == 0:
            return None
        key = self.key(block, height, width, dtype, device)
        if key in self.values:
            self.hits += 1
            return self.values[key]
        self.misses += 1
        mask = target_device_mask(block, height, width, dtype, device)
        if (
            mask is None
        ):  # Guard for type checkers; shifted blocks always return a mask.
            raise RuntimeError("A shifted block produced no attention mask")
        self.values[key] = mask
        return mask

    def clear(self) -> None:
        self.values.clear()
        self.hits = 0
        self.misses = 0


def iter_swin_blocks(model) -> Iterator[object]:
    for stage in model.encoder.encoder.layers:
        yield from stage.blocks


def install_mask_method(model, method: MaskMethod) -> None:
    """Install one benchmark variant on every Swin block."""
    for block in iter_swin_blocks(model):
        block.get_attn_mask = MethodType(method, block)


def install_cached_masks(model, cache: ShiftMaskCache) -> None:
    def cached_method(block, height, width, dtype, device):
        return cache.get(block, height, width, dtype, device)

    install_mask_method(model, cached_method)


def restore_transformers_masks(model) -> None:
    """Remove instance patches and restore the Transformers class method."""
    for block in iter_swin_blocks(model):
        if "get_attn_mask" in block.__dict__:
            del block.get_attn_mask
