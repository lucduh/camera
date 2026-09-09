# Shallow-draft speculative decoding

This is a separate Chapter 6 research branch. It tests speculative decoding
with a cheaper decoder that shares Donut's already-computed visual memory. It
is intentionally minimal and hackable; it is not integrated into the normal
Chapter 6 baselines yet.

## Idea

The target has four mBART decoder layers. A draft is made by copying the target
mBART decoder and retaining its first one or two layers. Both target and draft
receive the same encoder output. The draft proposes up to `K` tokens; the full
target verifies them in one causal forward. The longest matching prefix is
accepted and the target's correction/bonus token is emitted.

The draft is truncated first, then optionally distilled. No second vision
encoder pass is performed. Rejected target cache entries are cropped using
`EncoderDecoderCache.crop`; this crops self-attention history while preserving
encoder cross-attention state. This cache path must be checked on the pinned
Transformers version before trusting GPU measurements.

## 0. CPU mechanics test

```bash
uv run pytest tests/test_shallow_draft.py -q
```

The local CPU test uses a tiny two-layer Donut-like model and checks draft
construction, masked distillation loss, exact target-greedy agreement, and
proposal accounting. It does not establish useful speed.

## 1. Truncated-draft probe

First test an untrained truncation. It is a feasibility probe only; low
agreement is expected and does not justify a final negative conclusion.

```bash
TASK=configs/tasks/br.toml
CHECKPOINT=/secure/path/to/br/best
TEST=/secure/path/to/frozen/br/test.json

uv run python scripts/ch06/shallow_draft_probe.py \
  --task "$TASK" --checkpoint "$CHECKPOINT" --data "$TEST" \
  --output results/ch06/draft-truncated-d1.json \
  --draft-depth 1 --max-draft 4 --max-new-tokens 80 \
  --limit 32 --preset sdpa --dtype bf16
```

Repeat with `--draft-depth 2`, `--max-draft 1,2,4,8` as separate runs, and
`--confidence 0.8` or another declared threshold. The script records only
numeric summaries: target-greedy exactness, draft/target top-1 agreement at the
prompt, draft calls, proposal acceptance, target-call token efficiency, and
latency. It never writes generated text, tokens, images, document IDs, or
private paths. Note that prompt top-1 agreement is only a cheap diagnostic, not
sequence-level draft quality.

Run on a fine-tuned checkpoint, never on the base model for the final claim.
The target attention preset is recorded and must stay fixed across comparisons.

## 2. Distill the shallow draft

Use the frozen target as teacher. The draft receives teacher-forced target
prefixes and learns a masked combination of teacher-logit KL and ground-truth
cross-entropy. The target encoder and decoder are frozen. This script saves only
the standalone mBART draft decoder and a numeric training record.

```bash
TRAIN=/secure/path/to/br/train-fit.json

uv run python scripts/ch06/distill_shallow_draft.py \
  --task "$TASK" --checkpoint "$CHECKPOINT" --train "$TRAIN" \
  --output checkpoints/ch06-draft-d1 \
  --record results/ch06/distill-d1.json \
  --draft-depth 1 --height 1920 --width 1440 \
  --max-length 80 --batch-size 1 --steps 2000 \
  --learning-rate 1e-4 --temperature 1.0 --ce-weight 0.5 \
  --dtype bf16
```

Repeat for depth 2. Use a validation split for selecting steps and temperature;
the minimal script currently uses the supplied training file and is intended as
a research scaffold, not a final training pipeline. Distillation must not use
test documents.

## 3. Probe the distilled draft

```bash
uv run python scripts/ch06/shallow_draft_probe.py \
  --task "$TASK" --checkpoint "$CHECKPOINT" --data "$TEST" \
  --draft-checkpoint checkpoints/ch06-draft-d1 \
  --output results/ch06/draft-distilled-d1-k4.json \
  --draft-depth 1 --max-draft 4 --max-new-tokens 80 \
  --limit 32 --preset sdpa --dtype bf16
```

The draft checkpoint must have the same vocabulary, hidden dimension, encoder
memory projection and tokenizer vocabulary as the target. The script rejects
incompatible dimensions. Compare distilled and truncated drafts at exactly the
same documents and target settings.

## 4. What to test before making a claim

For depth 1 and 2, and proposal lengths 1, 2, 4, and 8, record:

- draft-only latency and target latency;
- target calls and draft calls;
- proposed, accepted and rejected tokens;
- exact equality with target greedy;
- cache crop/rejection frequency;
- decoder-only and complete model latency;
- peak memory.

The relevant speed condition is not acceptance alone. A useful configuration
must satisfy approximately:

```text
draft proposal cost + target verification cost + crop overhead
    < vanilla target greedy cost
```

Run the target-only greedy baseline with the same checkpoint, resolution,
attention backend, precision, images and naturally terminating output budget.
A candidate that changes tokens is not a valid acceleration result until the
cause is diagnosed; report it as an exactness failure rather than silently
scoring a different output.

## Limitations of this scaffold

- Batch size is one.
- The draft is independently copied and does not share live parameters with the
target; it shares only the encoder output.
- Cache cropping is implemented for the pinned Transformers cache API and needs
H100 validation with the actual Donut checkpoint.
- The proposal confidence value uses maximum softmax probability and is not
calibrated.
- The distillation script has no scheduler, checkpoint selection, validation
loop, or distributed training.
- FA4/SDPA backend comparisons belong to Chapter 5; keep the backend fixed here.

Do not merge this branch into the main Chapter 6 implementation until the
exactness audit and GPU latency measurements show whether shallow drafts have
headroom in this short-output document regime.
