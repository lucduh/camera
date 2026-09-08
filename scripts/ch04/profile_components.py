import argparse
import statistics
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch

from donut_camera.model import DEFAULT_MODEL, load_bundle, resolve_device
from donut_camera.records import write_record
from donut_camera.tasks import load_task
from donut_camera.timing import benchmark


@dataclass
class ModuleTimer:
    name: str
    pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    pending: list[torch.cuda.Event] = field(default_factory=list)

    def before(self, _module, _inputs) -> None:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self.pending.append(start)

    def after(self, _module, _inputs, _output) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.pairs.append((self.pending.pop(), end))

    def reset(self) -> None:
        self.pairs.clear()
        self.pending.clear()

    def elapsed_ms(self) -> float:
        return sum(start.elapsed_time(end) for start, end in self.pairs)


def attach_timer(module, name: str, timers: list[ModuleTimer], handles: list) -> None:
    timer = ModuleTimer(name)
    timers.append(timer)
    handles.append(module.register_forward_pre_hook(timer.before))
    handles.append(module.register_forward_hook(timer.after))


def encode(model, pixels):
    with torch.inference_mode():
        return model.encoder(pixels, return_dict=True)


def decode(model, pixels, prompt, encoder_outputs, output_length):
    with torch.inference_mode():
        return model.generate(
            pixel_values=pixels,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=prompt,
            min_new_tokens=output_length,
            max_new_tokens=output_length,
            use_cache=True,
        )


def event_time(call) -> tuple[object, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def aggregate(records: list[dict], group: str) -> dict:
    names = records[0][group]
    output = {}
    for name in names:
        values = [record[group][name] for record in records]
        output[name] = {
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "raw_ms": values,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile Donut encoder stages and decoder components"
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-length", type=int, default=56)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if not device.startswith("cuda"):
        raise RuntimeError("Component profiling requires CUDA")
    task = load_task(args.task)
    bundle = load_bundle(
        args.checkpoint, task=task, device=device, dtype=args.dtype, training=False
    )
    bundle.set_resolution(args.height, args.width)
    model = bundle.model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pixels = torch.randn(
        args.batch_size,
        model.encoder.config.num_channels,
        args.height,
        args.width,
        device=device,
        dtype=bundle.dtype,
        generator=generator,
    )
    prompt = bundle.decoder_start_ids(args.batch_size)
    encode_call = partial(encode, model, pixels)

    for _ in range(args.warmups):
        encoder_outputs = encode_call()
        decode(model, pixels, prompt, encoder_outputs, args.output_length)
    torch.cuda.synchronize()

    encoder_uninstrumented = benchmark(
        encode_call, warmups=args.warmups, repetitions=args.repetitions
    )
    encoder_outputs = encode_call()
    decode_call = partial(
        decode, model, pixels, prompt, encoder_outputs, args.output_length
    )
    decoder_uninstrumented = benchmark(
        decode_call, warmups=args.warmups, repetitions=args.repetitions
    )
    del encoder_outputs

    encoder_timers: list[ModuleTimer] = []
    decoder_timers: list[ModuleTimer] = []
    handles = []
    attach_timer(model.encoder.embeddings, "patch_embedding", encoder_timers, handles)
    for index, stage in enumerate(model.encoder.encoder.layers):
        attach_timer(stage, f"stage_{index}", encoder_timers, handles)
    attach_timer(
        model.decoder.model.decoder.embed_tokens,
        "token_embedding",
        decoder_timers,
        handles,
    )
    for index, layer in enumerate(model.decoder.model.decoder.layers):
        attach_timer(layer, f"layer_{index}", decoder_timers, handles)
    attach_timer(
        model.decoder.lm_head, "vocabulary_projection", decoder_timers, handles
    )

    records = []
    try:
        for _ in range(args.repetitions):
            for timer in encoder_timers + decoder_timers:
                timer.reset()
            encoder_outputs, encoder_total_ms = event_time(encode_call)
            decode_call = partial(
                decode, model, pixels, prompt, encoder_outputs, args.output_length
            )
            sequences, decoder_total_ms = event_time(decode_call)
            encoder_components = {
                timer.name: timer.elapsed_ms() for timer in encoder_timers
            }
            decoder_components = {
                timer.name: timer.elapsed_ms() for timer in decoder_timers
            }
            records.append(
                {
                    "encoder_total_ms": encoder_total_ms,
                    "decoder_total_ms": decoder_total_ms,
                    "generated_tokens": sequences.shape[1] - prompt.shape[1],
                    "encoder": encoder_components,
                    "decoder": decoder_components,
                    "encoder_unaccounted_ms": encoder_total_ms
                    - sum(encoder_components.values()),
                    "decoder_unaccounted_ms": decoder_total_ms
                    - sum(decoder_components.values()),
                }
            )
    finally:
        for handle in handles:
            handle.remove()

    path = write_record(
        args.output,
        config={
            "task": task.name,
            "checkpoint": Path(args.checkpoint).name,
            "height": args.height,
            "width": args.width,
            "batch_size": args.batch_size,
            "output_length": args.output_length,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": device,
            "attention": model.decoder.config._attn_implementation,
        },
        measurements={
            "uninstrumented": {
                "encoder": encoder_uninstrumented,
                "decoder": decoder_uninstrumented,
            },
            "encoder_components": aggregate(records, "encoder"),
            "decoder_components": aggregate(records, "decoder"),
            "raw_profiles": records,
        },
    )
    print(path)


if __name__ == "__main__":
    main()
