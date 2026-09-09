import argparse
import gc
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from transformers.models.donut.modeling_donut_swin import DonutSwinLayer

from donut_camera.model import DEFAULT_MODEL, load_bundle, resolve_device
from donut_camera.records import write_record
from donut_camera.swin_mask import (
    ShiftMaskCache,
    historical_mask,
    install_cached_masks,
    install_mask_method,
    restore_transformers_masks,
)
from donut_camera.tasks import load_task
from donut_camera.timing import benchmark, cuda_memory


def parse_resolutions(value: str) -> list[tuple[int, int]]:
    return [tuple(map(int, item.split("x"))) for item in value.split(",")]


def encode(model, pixels):
    with torch.inference_mode():
        return model.encoder(pixels, return_dict=True)


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    difference = (reference.float() - candidate.float()).abs()
    return {
        "exact": torch.equal(reference, candidate),
        "max_absolute_difference": difference.max().item(),
    }


def stage_shapes(model, image_height: int, image_width: int) -> list[dict]:
    height = image_height // model.encoder.config.patch_size
    width = image_width // model.encoder.config.patch_size
    stages = []
    for index, stage in enumerate(model.encoder.encoder.layers):
        shifted = [block for block in stage.blocks if block.shift_size > 0]
        representative = shifted[0]
        windows = (height // representative.window_size) * (
            width // representative.window_size
        )
        stages.append(
            {
                "stage": index,
                "height": height,
                "width": width,
                "window_size": representative.window_size,
                "shift_size": representative.shift_size,
                "windows": windows,
                "mask_shape": [
                    windows,
                    representative.window_size**2,
                    representative.window_size**2,
                ],
                "mask_elements": windows * representative.window_size**4,
                "shifted_blocks": len(shifted),
            }
        )
        if index < len(model.encoder.encoder.layers) - 1:
            height //= 2
            width //= 2
    return stages


def isolated_stage_profile(
    stage: dict,
    *,
    dtype: torch.dtype,
    device: torch.device,
    warmups: int,
    repetitions: int,
) -> dict:
    block = SimpleNamespace(
        window_size=stage["window_size"], shift_size=stage["shift_size"]
    )
    arguments = (stage["height"], stage["width"], dtype, device)
    current_call = partial(DonutSwinLayer.get_attn_mask, block, *arguments)
    historical_call = partial(historical_mask, block, *arguments)
    cache = ShiftMaskCache()
    cached_call = partial(cache.get, block, *arguments)

    current_timing = benchmark(current_call, warmups=warmups, repetitions=repetitions)
    historical_timing = benchmark(
        historical_call, warmups=warmups, repetitions=repetitions
    )

    def cold_call():
        cache.clear()
        return cached_call()

    cached_cold_timing = benchmark(cold_call, warmups=warmups, repetitions=repetitions)
    cache.clear()
    first_cached = cached_call()
    cached_hit_timing = benchmark(cached_call, warmups=warmups, repetitions=repetitions)
    cache_stats = {
        "entries": len(cache.values),
        "hits": cache.hits,
        "misses": cache.misses,
    }

    current = current_call()
    historical = historical_call()
    if first_cached is None or current is None or historical is None:
        raise RuntimeError("A shifted block unexpectedly returned no mask")
    correctness = {
        "historical_vs_current": compare(current, historical),
        "cached_vs_current": compare(current, first_cached),
        "cache_reuses_storage": first_cached.data_ptr() == cached_call().data_ptr(),
    }
    del current, historical, first_cached
    cache.clear()
    memory = {
        "historical": cuda_memory(historical_call),
        "current": cuda_memory(current_call),
        "cached_cold": cuda_memory(cold_call),
    }
    cache.clear()
    cached_call()
    memory["cached_hit"] = cuda_memory(cached_call)

    return {
        **stage,
        "mask_bytes": stage["mask_elements"]
        * torch.empty((), dtype=dtype).element_size(),
        "timing": {
            "historical": historical_timing,
            "current": current_timing,
            "cached_cold": cached_cold_timing,
            "cached_hit": cached_hit_timing,
        },
        "cuda_memory": memory,
        "correctness": correctness,
        "cache": cache_stats,
    }


def encoder_correctness(reference, candidate) -> dict:
    return compare(reference.last_hidden_state, candidate.last_hidden_state)


def integrated_encoder_profile(
    model,
    pixels: torch.Tensor,
    *,
    warmups: int,
    repetitions: int,
) -> dict:
    call = partial(encode, model, pixels)
    cache = ShiftMaskCache()
    try:
        # Check the complete encoder before timing. Release all outputs so each
        # memory benchmark starts from the same persistent model allocation.
        restore_transformers_masks(model)
        reference = call()
        install_mask_method(model, historical_mask)
        historical = call()
        historical_correctness = encoder_correctness(reference, historical)
        install_cached_masks(model, cache)
        cache.clear()
        cached = call()
        cached_correctness = encoder_correctness(reference, cached)
        visual_tokens = reference.last_hidden_state.shape[1]
        del reference, historical, cached
        cache.clear()
        torch.cuda.synchronize(pixels.device)

        restore_transformers_masks(model)
        current_timing = benchmark(call, warmups=warmups, repetitions=repetitions)
        current_memory = cuda_memory(call)

        install_mask_method(model, historical_mask)
        historical_timing = benchmark(call, warmups=warmups, repetitions=repetitions)
        historical_memory = cuda_memory(call)

        install_cached_masks(model, cache)

        def cached_cold_call():
            cache.clear()
            return call()

        cached_cold_timing = benchmark(
            cached_cold_call, warmups=warmups, repetitions=repetitions
        )
        cache.clear()
        cached_cold_memory = cuda_memory(cached_cold_call)
        cache.clear()
        call()
        cache_hits_before = cache.hits
        cached_warm_timing = benchmark(call, warmups=warmups, repetitions=repetitions)
        shifted_blocks = sum(
            block.shift_size > 0
            for stage in model.encoder.encoder.layers
            for block in stage.blocks
        )
        cache_stats = {
            "entries": len(cache.values),
            "misses": cache.misses,
            "hits_during_measured_warm_passes": cache.hits
            - cache_hits_before
            - warmups * shifted_blocks,
            "total_hits_before_memory_measurement": cache.hits,
        }
        cached_warm_memory = cuda_memory(call)
        return {
            "timing": {
                "historical": historical_timing,
                "current": current_timing,
                "cached_cold": cached_cold_timing,
                "cached_warm": cached_warm_timing,
            },
            "cuda_memory": {
                "historical": historical_memory,
                "current": current_memory,
                "cached_cold": cached_cold_memory,
                "cached_warm": cached_warm_memory,
            },
            "correctness": {
                "historical_vs_current": historical_correctness,
                "cached_vs_current": cached_correctness,
            },
            "cache": cache_stats,
            "visual_tokens": visual_tokens,
        }
    finally:
        restore_transformers_masks(model)
        cache.clear()


def estimated_mask_time_per_encoder(stage_records: list[dict]) -> dict:
    totals = {
        name: 0.0 for name in ("historical", "current", "cached_cold", "cached_warm")
    }
    for stage in stage_records:
        blocks = stage["shifted_blocks"]
        timing = stage["timing"]
        totals["historical"] += timing["historical"]["median_ms"] * blocks
        totals["current"] += timing["current"]["median_ms"] * blocks
        totals["cached_cold"] += timing["cached_cold"]["median_ms"]
        totals["cached_cold"] += timing["cached_hit"]["median_ms"] * (blocks - 1)
        totals["cached_warm"] += timing["cached_hit"]["median_ms"] * blocks
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce and profile DonutSwin shifted-window mask construction"
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", default="1280x960,1920x1440,2560x1920")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device")
    args = parser.parse_args()

    device_name = resolve_device(args.device)
    if not device_name.startswith("cuda"):
        raise RuntimeError("The reportable mask benchmark requires CUDA")
    device = torch.device(device_name)
    task = load_task(args.task)
    bundle = load_bundle(
        args.checkpoint,
        task=task,
        device=device_name,
        dtype=args.dtype,
        training=False,
    )
    model = bundle.model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    records = []

    for height, width in parse_resolutions(args.resolutions):
        bundle.set_resolution(height, width)
        shapes = stage_shapes(model, height, width)
        try:
            isolated = [
                isolated_stage_profile(
                    stage,
                    dtype=bundle.dtype,
                    device=device,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                )
                for stage in shapes
            ]
            pixels = torch.randn(
                args.batch_size,
                model.encoder.config.num_channels,
                height,
                width,
                dtype=bundle.dtype,
                device=device,
                generator=generator,
            )
            integrated = integrated_encoder_profile(
                model,
                pixels,
                warmups=args.warmups,
                repetitions=args.repetitions,
            )
            records.append(
                {
                    "status": "ok",
                    "resolution": [height, width],
                    "isolated_stages": isolated,
                    "estimated_mask_ms_per_encoder": estimated_mask_time_per_encoder(
                        isolated
                    ),
                    "integrated_encoder": integrated,
                }
            )
            del pixels
        except torch.OutOfMemoryError as error:
            records.append(
                {
                    "status": "oom",
                    "resolution": [height, width],
                    "error": str(error),
                }
            )
        finally:
            restore_transformers_masks(model)
            gc.collect()
            torch.cuda.empty_cache()

    path = write_record(
        args.output,
        config={
            "task": task.name,
            "checkpoint": Path(args.checkpoint).name,
            "resolutions": args.resolutions,
            "batch_size": args.batch_size,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": device_name,
            "attention": model.decoder.config._attn_implementation,
            "historical_transformers": "4.37.2 mask implementation",
            "current_transformers": transformers.__version__,
        },
        measurements={"records": records},
    )
    print(path)


if __name__ == "__main__":
    main()
