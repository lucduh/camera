import torch

from donut_camera.attention import apply_attention
from donut_camera.attention.experiments import (
    config,
    difference,
    load,
    parser,
    real_batches,
    synthetic,
)
from donut_camera.records import write_record


def observe(bundle, pixels, labels, *, max_new_tokens, gradients):
    model = bundle.model.eval()
    model.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(gradients):
        states = model.encoder(pixels, return_dict=True)
        output = model(encoder_outputs=states, labels=labels)
        result = {
            "encoder": states.last_hidden_state.detach().cpu(),
            "logits": output.logits.detach().cpu(),
            "loss": output.loss.item(),
        }
        if gradients:
            output.loss.backward()
            result["gradients"] = {
                name: p.grad.detach().cpu().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
    model.zero_grad(set_to_none=True)
    with torch.inference_mode():
        result["sequence"] = model.generate(
            pixel_values=pixels,
            decoder_input_ids=bundle.decoder_start_ids(pixels.shape[0]),
            max_new_tokens=max_new_tokens,
        ).cpu()
    return result


def main():
    cli = parser("Compare identical weights and inputs against eager; no text is saved")
    cli.add_argument("--data", type=str)
    cli.add_argument("--batches", type=int, default=4)
    cli.add_argument("--max-new-tokens", type=int, default=80)
    cli.add_argument("--gradients", action="store_true")
    args = cli.parse_args()
    bundle = load(args, baseline=True)
    source = (
        real_batches(bundle, args)
        if args.data
        else [{"pixel_values": p, "labels": l} for p, l in [synthetic(bundle, args)]]
    )
    rows = []
    for index, batch in enumerate(source):
        if index >= args.batches:
            break
        pixels = batch["pixel_values"].to(device=bundle.device, dtype=bundle.dtype)
        labels = batch["labels"].to(bundle.device)
        apply_attention(bundle.model, "baseline")
        reference = observe(
            bundle,
            pixels,
            labels,
            max_new_tokens=args.max_new_tokens,
            gradients=args.gradients,
        )
        apply_attention(bundle.model, args.preset)
        candidate = observe(
            bundle,
            pixels,
            labels,
            max_new_tokens=args.max_new_tokens,
            gradients=args.gradients,
        )
        row = {
            "batch": index,
            "encoder": difference(reference["encoder"], candidate["encoder"]),
            "logits": difference(reference["logits"], candidate["logits"]),
            "loss_absolute_error": abs(reference["loss"] - candidate["loss"]),
            "exact_sequence": torch.equal(reference["sequence"], candidate["sequence"]),
        }
        if args.gradients:
            row["gradients"] = {
                name: difference(grad, candidate["gradients"][name])
                for name, grad in reference["gradients"].items()
            }
        rows.append(row)
        write_record(args.output, config=config(args), measurements=rows)


if __name__ == "__main__":
    main()
