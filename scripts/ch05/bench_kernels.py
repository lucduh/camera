from contextlib import nullcontext
from functools import partial

import torch
from torch.nn.attention import sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from donut_camera.attention import BACKENDS
from donut_camera.attention.experiments import config, difference, load, parser
from donut_camera.records import write_record
from donut_camera.timing import benchmark, cuda_memory


def attention(q, k, v, mask, causal, backend):
    if backend == "fa":
        if mask is not None:
            raise NotImplementedError("FA4 adapter does not support additive Swin bias")
        from flash_attn.cute import flash_attn_func

        return flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal
        ).transpose(1, 2)
    context = nullcontext() if backend == "auto" else sdpa_kernel([BACKENDS[backend]])
    with context:
        return scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)


@torch.inference_mode()
def main():
    cli = parser("Kernel sweep with actual Donut head shapes and Swin biases")
    cli.add_argument("--kv-lengths", default="1,16,56,80,512,2048")
    cli.add_argument("--backends", default="auto,math,efficient,flash,cudnn,fa")
    cli.add_argument("--warmups", type=int, default=5)
    cli.add_argument("--repetitions", type=int, default=20)
    args = cli.parse_args()
    bundle = load(args, baseline=True)
    if bundle.device.type != "cuda":
        raise RuntimeError("Kernel measurements require CUDA")
    model = bundle.model
    heads = model.decoder.config.decoder_attention_heads
    dim = model.decoder.config.d_model // heads
    shapes = []
    for length in map(int, args.kv_lengths.split(",")):
        shapes.extend(
            [
                ("decode", args.batch_size, heads, dim, 1, length, None, False),
                ("prefill", args.batch_size, heads, dim, length, length, None, True),
            ]
        )
    shapes.append(
        (
            "cross",
            args.batch_size,
            heads,
            dim,
            1,
            args.height // 32 * (args.width // 32),
            None,
            False,
        )
    )
    height, width = args.height // 4, args.width // 4
    for index, stage in enumerate(model.encoder.encoder.layers):
        for shifted in (False, True):
            block = stage.blocks[1 if shifted else 0]
            module = block.attention.self
            window = block.window_size
            h, w = (
                (height + window - 1) // window * window,
                (width + window - 1) // window * window,
            )
            windows = h // window * (w // window)
            length = window**2
            bias = (
                module.relative_position_bias_table[
                    module.relative_position_index.flatten()
                ]
                .view(length, length, -1)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            shift = block.get_attn_mask(h, w, bundle.dtype, bundle.device)
            if shift is not None:
                bias = bias + shift.repeat(args.batch_size, 1, 1).unsqueeze(1)
            shapes.append(
                (
                    f"swin_{index}_{'shifted' if shifted else 'plain'}",
                    args.batch_size * windows,
                    module.num_attention_heads,
                    module.attention_head_size,
                    length,
                    length,
                    bias,
                    False,
                )
            )
        height, width = (height + 1) // 2, (width + 1) // 2
    records = []
    for regime, batch, heads, dim, qlen, klen, bias, causal in shapes:
        q = torch.randn(
            batch, heads, qlen, dim, device=bundle.device, dtype=bundle.dtype
        )
        k = torch.randn(
            batch, heads, klen, dim, device=bundle.device, dtype=bundle.dtype
        )
        v = torch.randn_like(k)
        reference = attention(q, k, v, bias, causal, "math")
        for backend in args.backends.split(","):
            row = {
                "regime": regime,
                "backend": backend,
                "shape": [batch, heads, qlen, klen, dim],
            }
            call = partial(attention, q, k, v, bias, causal, backend)
            try:
                row.update(
                    status="ok",
                    error=difference(reference, call()),
                    timing=benchmark(
                        call, warmups=args.warmups, repetitions=args.repetitions
                    ),
                    memory=cuda_memory(call),
                )
            except (RuntimeError, ImportError, NotImplementedError) as error:
                row.update(
                    status="unavailable_or_failed", error_type=type(error).__name__
                )
            records.append(row)
            write_record(args.output, config=config(args), measurements=records)


if __name__ == "__main__":
    main()
