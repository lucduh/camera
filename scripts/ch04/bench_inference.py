import argparse
import gc
from functools import partial
from pathlib import Path

import torch

from donut_camera.model import DEFAULT_MODEL, load_bundle, resolve_device
from donut_camera.records import write_record
from donut_camera.tasks import load_task
from donut_camera.timing import benchmark, cuda_memory


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_resolutions(value: str) -> list[tuple[int, int]]:
    return [tuple(map(int, item.split("x"))) for item in value.split(",")]


def encode_model(model, pixels):
    with torch.inference_mode():
        return model.encoder(pixels, return_dict=True)


def generate_model(model, pixels, prompt, output_length, encoder_outputs=None):
    with torch.inference_mode():
        return model.generate(
            pixel_values=pixels,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=prompt,
            min_new_tokens=output_length,
            max_new_tokens=output_length,
            use_cache=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled Donut inference scaling benchmark"
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", default="1280x960,1920x1440,2560x1920")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--output-lengths", default="16,32,56,64,80")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device")
    args = parser.parse_args()

    task = load_task(args.task)
    device = resolve_device(args.device)
    if not device.startswith("cuda"):
        raise RuntimeError("The reportable inference benchmark requires CUDA")
    bundle = load_bundle(
        args.checkpoint, task=task, device=device, dtype=args.dtype, training=False
    )
    model = bundle.model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    records = []

    for height, width in parse_resolutions(args.resolutions):
        bundle.set_resolution(height, width)
        for batch_size in parse_ints(args.batch_sizes):
            pixels = torch.randn(
                batch_size,
                model.encoder.config.num_channels,
                height,
                width,
                device=device,
                dtype=bundle.dtype,
                generator=generator,
            )
            prompt = bundle.decoder_start_ids(batch_size)
            encode = partial(encode_model, model, pixels)

            try:
                encoder_timing = benchmark(
                    encode, warmups=args.warmups, repetitions=args.repetitions
                )
                encoder_memory = cuda_memory(encode)
                with torch.inference_mode():
                    encoder_outputs = encode()
                visual_tokens = encoder_outputs.last_hidden_state.shape[1]

                for output_length in parse_ints(args.output_lengths):
                    decode = partial(
                        generate_model,
                        model,
                        pixels,
                        prompt,
                        output_length,
                        encoder_outputs,
                    )
                    full_inference = partial(
                        generate_model, model, pixels, prompt, output_length
                    )

                    decoder_timing = benchmark(
                        decode, warmups=args.warmups, repetitions=args.repetitions
                    )
                    decoder_memory = cuda_memory(decode)
                    del encoder_outputs
                    gc.collect()
                    torch.cuda.empty_cache()
                    full_timing = benchmark(
                        full_inference,
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                    )
                    full_memory = cuda_memory(full_inference)
                    records.append(
                        {
                            "status": "ok",
                            "resolution": [height, width],
                            "batch_size": batch_size,
                            "output_length": output_length,
                            "visual_tokens": visual_tokens,
                            "encoder": encoder_timing,
                            "decoder": decoder_timing,
                            "full": full_timing,
                            "memory": {
                                "encoder": encoder_memory,
                                "decoder": decoder_memory,
                                "full": full_memory,
                            },
                        }
                    )
                    # Recreate the persistent states for the next decoder length.
                    with torch.inference_mode():
                        encoder_outputs = encode()
                del encoder_outputs
            except torch.OutOfMemoryError as error:
                records.append(
                    {
                        "status": "oom",
                        "resolution": [height, width],
                        "batch_size": batch_size,
                        "error": str(error),
                    }
                )
            finally:
                del pixels, prompt
                gc.collect()
                torch.cuda.empty_cache()

    path = write_record(
        args.output,
        config={
            "task": task.name,
            "checkpoint": Path(args.checkpoint).name,
            "resolutions": args.resolutions,
            "batch_sizes": args.batch_sizes,
            "output_lengths": args.output_lengths,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": device,
            "decoder_attention": model.decoder.config._attn_implementation,
            "mask_cache": False,
        },
        measurements={"records": records},
    )
    print(path)


if __name__ == "__main__":
    main()
