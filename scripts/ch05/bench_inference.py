from functools import partial

import torch

from donut_camera.attention import fallback_calls
from donut_camera.attention.experiments import config, load, parser, synthetic
from donut_camera.records import write_record
from donut_camera.timing import benchmark, cuda_memory


@torch.inference_mode()
def encode(model, pixels):
    return model.encoder(pixels, return_dict=True)


@torch.inference_mode()
def generate(model, pixels, prompt, length, states=None):
    kwargs = {} if states is None else {"encoder_outputs": states}
    return model.generate(
        pixel_values=pixels,
        decoder_input_ids=prompt,
        min_new_tokens=length,
        max_new_tokens=length,
        use_cache=True,
        **kwargs,
    )


def main():
    cli = parser("Chapter 5 controlled inference, one preset and shape per run")
    cli.add_argument("--warmups", type=int, default=5)
    cli.add_argument("--repetitions", type=int, default=20)
    args = cli.parse_args()
    bundle = load(args)
    if bundle.device.type != "cuda":
        raise RuntimeError("Reportable inference timing requires CUDA")
    pixels, labels = synthetic(bundle, args)
    del labels
    model = bundle.model.eval()
    prompt = bundle.decoder_start_ids(args.batch_size)
    measurements = {}
    encoder = partial(encode, model, pixels)
    states = None
    for phase in ("encoder", "decoder", "full"):
        if phase == "encoder":
            call = encoder
        elif phase == "decoder":
            states = encoder()
            call = partial(generate, model, pixels, prompt, args.length, states)
        else:
            # Release the partial as well as the local reference to visual states.
            del call
            states = None
            call = partial(generate, model, pixels, prompt, args.length)
        timing = benchmark(call, warmups=args.warmups, repetitions=args.repetitions)
        measurements[phase] = {
            **timing,
            "memory": cuda_memory(call),
            "documents_per_second": args.batch_size * 1000 / timing["median_ms"],
        }
        if phase != "encoder":
            measurements[phase]["tokens_per_second"] = (
                args.batch_size * args.length * 1000 / timing["median_ms"]
            )
    measurements["cudnn_fallback_calls_including_warmups"] = fallback_calls(model)
    write_record(args.output, config=config(args), measurements=measurements)


if __name__ == "__main__":
    main()
