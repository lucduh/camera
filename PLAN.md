# Donut Camera experiment plan

## Rules

- BR is the primary task; KPID and KPD provide coverage.
- Standard resolutions are 1280x960, 1920x1440, and 2560x1920.
- Use explicit train, validation, and test splits.
- Save raw measurements and environment metadata in JSON.
- Keep predictions and identifiable paths out of reportable records.
- Record unsupported configurations and OOMs; never silently fall back.
- Stabilize each chapter before implementing the next one.

## Foundation

- [x] Task-specific schemas
- [x] Dataset loading and fixed-schema target serialization
- [x] Data/schema audit command
- [x] Metrics in which a wrong value contributes FP and FN
- [x] Minimal model loading and resolution configuration
- [x] Explicit-split training command
- [x] Evaluation command with sanitized per-document records
- [ ] Verify field spellings against all three real datasets
- [ ] Freeze validation splits
- [ ] Audit untruncated target lengths
- [ ] Complete a BR smoke train/save/reload/evaluate run
- [ ] Freeze canonical fine-tuning hyperparameters

## Chapter 4 — Computational characterization

1. Architecture, parameter, token, window, and target-length inventory.
2. Baseline fine-tuning and quality at the three resolution tiers.
3. Real batch-one inference decomposition: preprocessing, transfer, encoder,
   decoder, parsing, and wall time.
4. Controlled resolution x output-length inference sweep.
5. Batch-size latency, throughput, and memory sweep.
6. Encoder-stage and decoder-component CUDA-event profile.
7. Controlled training-step sweep: forward, backward, optimizer, and memory.
8. Real training-pipeline profile including DataLoader wait and transfer.
9. Historical/current/cached shifted-window mask reproduction.

BR receives the complete grid. KPID and KPD receive baseline quality/training,
real inference, and selected low/high-resolution confirmation runs.

## Chapter 5 — Attention acceleration

1. Implement mask caching, encoder SDPA, decoder SDPA controls, and optional FA4
   as independent switches.
2. Audit encoder outputs, logits, sequences, losses, and gradients.
3. Benchmark Swin-window, decode, and prefill/training kernel shapes.
4. Benchmark incremental full-model configurations.
5. Compare training-step latency and memory.
6. Run high-resolution BF16 stability ablations.

## Chapter 6 — Structured decoding

1. Implement a custom greedy KV-cache loop with exact Hugging Face parity.
2. Build a minimal ordered-field schema from each task configuration.
3. Evaluate constrained/template decoding.
4. Implement and evaluate a clearly defined speculative baseline.
5. Implement exact-verification and forced schema-aware block modes.
6. Report model calls, forced/proposed/accepted tokens, latency, validity, and F1
   on BR, with KPID/KPD confirmation.

## Completion gates

A benchmark result is reportable only if:

- its model, task, data split, resolution, batch, length, dtype, software, GPU,
  seed, and Git revision are recorded;
- CUDA timings are synchronized and include raw repetitions;
- memory distinguishes starting allocation from incremental peak;
- correctness has been checked at the level required by the experiment;
- the canonical command is documented;
- the analysis reads saved records rather than copied console output.
