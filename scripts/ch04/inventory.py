import argparse
import math
from pathlib import Path

from donut_camera.model import DEFAULT_MODEL, load_bundle
from donut_camera.records import write_record
from donut_camera.tasks import load_task


def parse_resolutions(value: str) -> list[tuple[int, int]]:
    resolutions = []
    for item in value.split(","):
        height, width = item.strip().split("x")
        resolutions.append((int(height), int(width)))
    return resolutions


def parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def architecture(bundle, resolutions: list[tuple[int, int]]) -> dict:
    model = bundle.model
    config = model.encoder.config
    patch_size = config.patch_size
    if isinstance(patch_size, list | tuple):
        patch_height, patch_width = patch_size
    else:
        patch_height = patch_width = patch_size
    window_size = config.window_size
    depths = list(config.depths)
    embed_dim = config.embed_dim

    resolution_records = []
    for image_height, image_width in resolutions:
        height = image_height // patch_height
        width = image_width // patch_width
        stages = []
        total_block_macs = 0
        for index, depth in enumerate(depths):
            dimension = embed_dim * 2**index
            tokens = height * width
            windows = math.ceil(height / window_size) * math.ceil(width / window_size)
            padded_tokens = windows * window_size**2
            # Approximation for QKV/output, 4x MLP, and QK/AV operations.
            dense_macs = depth * 12 * tokens * dimension**2
            attention_macs = depth * 2 * padded_tokens * window_size**2 * dimension
            block_macs = dense_macs + attention_macs
            total_block_macs += block_macs
            stages.append(
                {
                    "stage": index,
                    "grid": [height, width],
                    "tokens": tokens,
                    "padded_window_tokens": padded_tokens,
                    "windows": windows,
                    "depth": depth,
                    "dimension": dimension,
                    "estimated_block_macs": block_macs,
                }
            )
            if index < len(depths) - 1:
                height = math.ceil(height / 2)
                width = math.ceil(width / 2)
        resolution_records.append(
            {
                "image": [image_height, image_width],
                "final_visual_tokens": height * width,
                "estimated_encoder_block_macs": total_block_macs,
                "stages": stages,
            }
        )

    encoder_stages = model.encoder.encoder.layers
    decoder_layers = model.decoder.model.decoder.layers
    return {
        "parameters": {
            "model": parameters(model),
            "encoder": parameters(model.encoder),
            "decoder": parameters(model.decoder),
            "encoder_stages": [parameters(stage) for stage in encoder_stages],
            "decoder_layers": [parameters(layer) for layer in decoder_layers],
            "lm_head": parameters(model.decoder.lm_head),
        },
        "encoder_config": {
            "patch_size": [patch_height, patch_width],
            "window_size": window_size,
            "depths": depths,
            "embed_dim": embed_dim,
            "num_heads": list(config.num_heads),
        },
        "resolutions": resolution_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Donut architecture inventory")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", default="1280x960,1920x1440,2560x1920")
    args = parser.parse_args()

    task = load_task(args.task)
    bundle = load_bundle(
        args.checkpoint, task=task, device="cpu", dtype="fp32", training=False
    )
    measurements = architecture(bundle, parse_resolutions(args.resolutions))
    path = write_record(
        args.output,
        config={"task": task.name, "checkpoint": Path(args.checkpoint).name},
        measurements=measurements,
    )
    print(path)
    print(measurements)


if __name__ == "__main__":
    main()
