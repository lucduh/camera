# Chapter 6: structured decoding baselines

Run from `donut-camera/`. Initial scope is BR, batch one, one-token task prompt,
a fine-tuned checkpoint, and a fixed attention preset. No model training or
separate draft model is required. GPU measurements have not been run locally.

```bash
TASK=configs/tasks/br.toml
CHECKPOINT=/secure/path/to/best
DATA=/secure/path/to/frozen-audit.json
mkdir -p results/ch06
```

## Implemented methods

- `hf`: Hugging Face greedy reference with actual decoder-forward counting.
- `greedy`: explicit cached loop.
- `template`: force field opening tags, generate values until the expected close
  tag, and force final EOS. Consume the previous uncached token and opening tag
  in the same forward. Unexpected EOS stops immediately; exhaustion without an
  expected close is reported, not repaired. Other unexpected tags remain model
  output and are assessed by the common validity/quality evaluator.
- `speculative`: schema proposals, optionally including missing-value chains;
  causal verification accepts the longest matching prefix plus a correction or
  bonus token, subject to EOS and the remaining budget. On rejection, rebuild
  the accepted-prefix cache excluding its still-uncached last token. Rebuild
  forwards count toward total calls. No hidden cache-cropping optimization.

`max_new_tokens` counts every emitted token, including forced tags and EOS, but
not the initial task token. Forced-token counts include template insertions and
forced BOS/EOS generation rules. Target-call counts come from actual forwards,
not emitted tokens or loop iterations. `max_draft=0` is a greedy control.

### Explicit generation policy

All four paths use the same newly constructed greedy policy: no sampling, one
beam, KV cache, checkpoint start/pad/EOS and forced BOS/EOS settings, and the
specified output budget. Other checkpoint generation processors (such as
repetition penalties, suppression lists or minimum-length rules) are deliberately
not inherited. Thus parity means parity with **HF under this declared policy**,
not arbitrary `generate()` settings. The policy is recorded, does not mutate the
checkpoint config, and uses pinned Transformers logits processors for forced
BOS/EOS. A different policy requires extending and re-auditing all paths.

The template intentionally overrides structural choices and is not required to
match greedy. Speculative verification is intended to preserve greedy, but
multi-position versus single-position floating-point execution can change
argmax decisions. Real-checkpoint equality is therefore a mandatory gate, not
an assumption inferred from toy tests.

## 1. Exactness and quality audit

```bash
uv run python scripts/ch06/audit_decoding.py \
  --task "$TASK" --checkpoint "$CHECKPOINT" --data "$DATA" \
  --preset sdpa --height 1920 --width 1440 --dtype bf16 \
  --max-new-tokens 80 --max-draft 8 \
  --output results/ch06/audit-k8.json
```

For a CPU smoke test use `--device cpu --dtype fp32 --limit 1` and a small
budget. For canonical quality, use the complete frozen test set with the full
80-token budget. All four strategies see the same encoder outputs. Records
include exact sequence agreement, per-document mechanism counters, and aggregate
strict/normalized extraction metrics and structured validity. Greedy or
speculative mismatch saves the audit record and exits nonzero. Template mismatch
is expected to be possible and does not fail the gate.

No images, text, field values, token IDs, document identifiers or private paths
are written into these records. Numeric per-document rows use source-order
indices. Keep even aggregate experiment records private unless approved.

## 2. Paired latency benchmark

```bash
uv run python scripts/ch06/bench_decoding.py \
  --task "$TASK" --checkpoint "$CHECKPOINT" --data "$DATA" \
  --preset sdpa --height 1920 --width 1440 --dtype bf16 \
  --max-new-tokens 80 --max-draft 8 \
  --warmups 3 --repetitions 10 \
  --output results/ch06/bench-k8.json
```

Requires CUDA. For each document and strategy it measures:

- Decoder-only latency with persistent precomputed encoder outputs.
- Pixel-to-sequence model latency, with a fresh encoder pass on every repetition.
- Peak allocated memory, raw synchronized timings and median/p95 summaries.
- Target calls, emitted/forced/proposed/accepted tokens and cache rebuilds.
- Exactness and extraction quality using the same parser/metrics as other chapters.

Model latency excludes image loading, preprocessing, transfer and parsing; do not
label it complete deployment latency. Per-document medians are aggregated across
documents; the summary p95 is the p95 of those medians, not pooled repetitions.
Method order rotates across documents. Decoder counting remains enabled for all
measurements (HF uses a lightweight forward hook). Parsing and equality checks
are outside timed regions. Greedy/speculative mismatches invalidate the timing
record and cause a nonzero exit. Runtime/backend/OOM failures propagate rather
than being silently counted as successful measurements.

Add `--profile-rebuilds` to perform a **separate** instrumented speculative pass
that synchronizes around cache reconstruction. Its `instrumented_rebuild_ms`
is separate from the primary latency samples. This diagnostic measures forward
reconstruction only, not all proposal/verification/host overhead.

## 3. Proposal ablation

Run both the audit and benchmark for each setting, retaining separate files:

```bash
for K in 1 2 4 8; do
  for CHAIN in on off; do
    EXTRA=()
    if [ "$CHAIN" = off ]; then EXTRA=(--no-missing-chain); fi
    uv run python scripts/ch06/audit_decoding.py \
      --task "$TASK" --checkpoint "$CHECKPOINT" --data "$DATA" \
      --preset sdpa --max-draft "$K" "${EXTRA[@]}" \
      --output "results/ch06/audit-k${K}-${CHAIN}.json" || exit 1
    uv run python scripts/ch06/bench_decoding.py \
      --task "$TASK" --checkpoint "$CHECKPOINT" --data "$DATA" \
      --preset sdpa --max-draft "$K" "${EXTRA[@]}" \
      --output "results/ch06/bench-k${K}-${CHAIN}.json" || exit 1
  done
done
```

The scripts share Chapter 5 setup arguments; `--length` is not used here (use
`--max-new-tokens`). No Chapter 6.4 block-decoding method or batched speculative
verification is implemented by this baseline work.

## Tests

```bash
uv run ruff check .
uv run pytest -q
```

Offline tests exercise real tiny Donut models for HF/greedy/speculative parity,
forced BOS/EOS and non-mutation of generation settings. Scripted models cover
full acceptance, rejection and cache reconstruction, early EOS, template forcing,
missing closing tags, short budgets, zero-proposal fallback and diagnostic timing.
Real BR and H100 gates remain necessary before reporting any speedup or exactness
claim from the research checkpoint.
