"""Chapter 5 presets. Apply to an eager-loaded model before any forward pass."""

from functools import partial

from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from donut_camera.attention.encoder_sdpa import apply_encoder_sdpa, revert_encoder_sdpa
from donut_camera.swin_mask import (
    ShiftMaskCache,
    install_cached_masks,
    restore_transformers_masks,
)

PRESETS = (
    "baseline",
    "eager",
    "encoder_sdpa",
    "sdpa",
    "sdpa_flash",
    "sdpa_efficient",
    "sdpa_math",
    "sdpa_cudnn",
    "fa",
)
BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


def restricted_attention(
    module, query, key, value, attention_mask, *, backend, **kwargs
):
    # Explicit shape-local fallback, with observable call counts. Never scope
    # this context around the encoder or the complete generation call.
    choices = [BACKENDS[backend]]
    if backend == "cudnn" and key.shape[-2] == 1:
        choices = [BACKENDS["efficient"]]
        module._ch05_fallback_calls = getattr(module, "_ch05_fallback_calls", 0) + 1
    with sdpa_kernel(choices):
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )


for _name in BACKENDS:
    _implementation = f"donut_camera_{_name}"
    AttentionInterface.register(
        _implementation, partial(restricted_attention, backend=_name)
    )
    AttentionMaskInterface.register(_implementation, AttentionMaskInterface["sdpa"])


def apply_attention(model, preset: str) -> None:
    if preset not in PRESETS:
        raise ValueError(f"Unknown attention preset: {preset}")
    restore_attention(model)
    decoder = "eager"
    if preset.startswith("sdpa_"):
        decoder = f"donut_camera_{preset.removeprefix('sdpa_')}"
    elif preset == "sdpa":
        decoder = "sdpa"
    elif preset == "fa":
        decoder = "flash_attention_4"
    model.decoder.set_attn_implementation(decoder)
    if preset != "baseline":
        model._ch05_mask_cache = ShiftMaskCache()
        install_cached_masks(model, model._ch05_mask_cache)
    if preset not in ("baseline", "eager"):
        apply_encoder_sdpa(model)
    model._ch05_preset = preset


def restore_attention(model) -> None:
    revert_encoder_sdpa(model)
    restore_transformers_masks(model)
    if hasattr(model, "_ch05_mask_cache"):
        model._ch05_mask_cache.clear()
        del model._ch05_mask_cache
    for module in model.decoder.modules():
        if hasattr(module, "_ch05_fallback_calls"):
            del module._ch05_fallback_calls
    model.decoder.set_attn_implementation("eager")
    model._ch05_preset = "baseline"


def fallback_calls(model) -> int:
    return sum(getattr(m, "_ch05_fallback_calls", 0) for m in model.decoder.modules())
