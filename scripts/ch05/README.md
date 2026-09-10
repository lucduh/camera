# Chapter 5 experiments

Run from `donut-camera/`. Use the same H100, frozen BR checkpoint, lockfile,
seed and inputs for every comparison. No canonical GPU results are included.

```bash
TASK=configs/tasks/br.toml
CHECKPOINT=/secure/path/to/best
DATA=/secure/path/to/frozen-audit.json
TRAIN=/secure/path/to/train-fit.json
mkdir -p results/ch05
```

Presets: `baseline` (unchanged eager), `eager` (cached masks), `encoder_sdpa`
(encoder SDPA, eager decoder), `sdpa` (automatic), `sdpa_flash`,
`sdpa_efficient`, `sdpa_math`, `sdpa_cudnn`, and `fa` (FA4 decoder).
All presets except baseline use cached masks. Forced restrictions apply only
to decoder attention. The encoder-only control separates encoder and decoder
gains. The cuDNN preset explicitly routes key length 1 to efficient attention;
it counts these calls. Other cuDNN failures are not silently swallowed. This
policy is an experimental control, not a claim about current cuDNN eligibility:
the kernel sweep also tests pure cuDNN.

FA4 is optional: `uv sync --extra fa4`. The working environment is pinned to
`flash-attn-4==4.0.0b17` and `nvidia-cutlass-dsl==4.5.2`; newer CUTLASS DSL
4.6 releases are incompatible with this FA4 beta. Check imports before a run:

```bash
uv run --extra fa4 python scripts/ch05/check_fa4.py
```

Its kernel compatibility and performance still require validation on H100.
Omit `fa` if unavailable; do not substitute a different backend under that name.

## 1. Kernel regimes

```bash
for BATCH in 1 2 4 8; do
  uv run python scripts/ch05/bench_kernels.py \
    --task "$TASK" --checkpoint "$CHECKPOINT" \
    --height 1920 --width 1440 --batch-size "$BATCH" \
    --kv-lengths 1,16,32,56,80,512,2048,4096 \
    --dtype bf16 --warmups 5 --repetitions 20 \
    --output "results/ch05/kernels-b${BATCH}.json"
done
```

Includes cached decode (q=1, noncausal over the existing cache), causal prefill,
visual cross-attention, and shifted/unshifted Swin windows with checkpoint
relative-position bias and actual stage head shapes. Pure kernels exclude
projection and mask-construction time. Swin additive-bias construction is outside
timing. FA4's adapter does not support that bias and explicitly marks it
unavailable. Failed/unsupported kernel cases have no latency value. Error status
alone does not prove backend ineligibility; inspect/reproduce failures before
reporting them as such. Math is the numerical reference, not ground truth.

## 2. Full-model inference

```bash
for PRESET in baseline eager encoder_sdpa sdpa sdpa_flash sdpa_efficient sdpa_math sdpa_cudnn fa; do
  for BATCH in 1 2 4 8; do
    uv run python scripts/ch05/bench_inference.py \
      --task "$TASK" --checkpoint "$CHECKPOINT" --preset "$PRESET" \
      --height 1920 --width 1440 --batch-size "$BATCH" --length 56 \
      --dtype bf16 --warmups 5 --repetitions 20 \
      --output "results/ch05/inference-${PRESET}-b${BATCH}.json"
  done
done
```

Repeat at 1280x960 and 2560x1920 using distinct output filenames. Every command
loads a fresh model. Encoder, cached-encoder decoder generation, and full
pixel-to-sequence generation are measured separately. These are synthetic fixed
length measurements, not quality measurements. Failures in model-level scripts
exit nonzero rather than produce misleading successful records.

## 3. Numerical audit and real quality

```bash
for PRESET in eager encoder_sdpa sdpa sdpa_math fa; do
  uv run python scripts/ch05/audit_attention.py \
    --task "$TASK" --checkpoint "$CHECKPOINT" --preset "$PRESET" \
    --data "$DATA" --batches 16 --batch-size 1 --length 80 \
    --max-new-tokens 80 --dtype bf16 \
    --output "results/ch05/audit-${PRESET}.json"
done
```

Repeat without `--data`, with `--batches 1 --gradients`, for a controlled
output/loss/gradient audit. Gradient snapshots are kept on CPU, not saved.
The gradient audit uses eval mode to remove dropout noise; it is distinct from
training-stability testing. Reports contain numeric errors and sequence equality,
never images, text, token IDs or document identifiers. No tolerance is silently
used to turn a numerical difference into a pass. Declare precision-specific
acceptance thresholds before interpreting results.

For strict/normalized F1 and validity, use the existing evaluator on exactly the
same frozen subset or complete test set for every preset, including baseline:

```bash
uv run python scripts/evaluate.py \
  --task "$TASK" --checkpoint "$CHECKPOINT" --data "$DATA" \
  --attention-backend sdpa --batch-size 1 --workers 0 --max-new-tokens 80 \
  --output results/ch05/quality-sdpa.json
```

## 4. Real-update training stability

```bash
for PRESET in baseline eager encoder_sdpa sdpa sdpa_efficient sdpa_flash sdpa_math sdpa_cudnn fa; do
  uv run python scripts/ch05/bench_stability.py \
    --task "$TASK" --checkpoint "$CHECKPOINT" --preset "$PRESET" \
    --data "$TRAIN" --height 2560 --width 1920 --length 80 \
    --batch-size 1 --dtype bf16 --steps 20 --learning-rate 3e-4 \
    --output "results/ch05/stability-${PRESET}-bf16.json"
done
```

Repeat automatic SDPA with `--dtype fp32`. Each process reloads the same weights
and seed; AdamW updates have nonzero learning rate. Real batches repeat in source
order if needed; without `--data`, the same synthetic batch repeats. Checks cover
loss, gradients, parameters and optimizer states after each update. No failure
within 20 steps is only evidence for that finite horizon, not general stability.

Repeat a failing case with `--diagnostic` and a distinct output filename to
identify non-finite leaf-module forward outputs. Diagnostic hooks synchronize
and perturb execution; their timing is explicitly marked instrumented. Regular
step timing excludes transfer and finite checks, includes forward/backward,
clipping and update, and summarizes steps after the first optimizer-state
allocation. CUDA peak memory includes the complete measured step.

Complete training also accepts `scripts/train.py --attention-backend PRESET`.
Saved weights do not serialize monkey patches: explicitly select the same preset
when reloading for evaluation. Chapter 4 defaults remain unchanged eager.

## Validation / interpretation

```bash
uv run ruff check .
uv run pytest -q
```

CPU tests cover eager/SDPA forward and gradient agreement, cached decoding versus
parallel causal decoding, and restoring the eager implementation. Forced GPU
backends and FA4 still need the target-machine audits before any performance or
stability claims from Chapter 5 become canonical. Keep result records and private
paths out of commits. These scripts do not change the report's existing findings.
