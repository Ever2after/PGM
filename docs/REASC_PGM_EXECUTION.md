# ReASC setting / PGM comparison execution

Started 2026-11-02 UTC. User authorized running on GPU4–7.

Live status: [progress](../artifacts/runs/reasc_pgm/progress.md), [machine status](../artifacts/runs/reasc_pgm/status.json).
This file describes the execution protocol, not completed performance results.

| Split | Questions | Responses/question | Purpose |
|---|---:|---:|---|
| Original MATH train | 800 | 16 | PGM selector, representation, scaler, continuation fitting |
| Original MATH train | 200 | 16 | Checkpoint and policy selection |
| Official calibration pool | 128 | 16 | ReASC uses first response only; PGM temperature uses all prefixes |
| Official MATH500 | 500 | 16 | Repeated benchmark, previously observed in earlier work |
| Previously unused original MATH test | 500 | 16 | Fresh scientific comparison |

Total 2,128 questions / 34,048 responses. Cohort selection was completed before generation, with normalized cross-cohort overlap0 and no historical overlap in the fresh cohort. The new cohort manifest includes source hashes and original train/test membership checks. No original locked datasets or results were overwritten.

Runtime allocation:

- GPU4,5: independent Gemma3-4B-IT vLLM TP1 replicas, bfloat16, max context8192, GPU memory fraction.25 each, up to128 sequences / eight questions per batch, CUDA decode graphs. The original serial16/TP2/eager startup runtime was replaced before policy locking after its throughput bottleneck was measured.
- GPU6: identical generated token IDs scored by Transformers, raw full-vocabulary confidence and final-layer full-reasoning embeddings.
- GPU7: pinned public xVerify-3B-Ia; raw-response and candidate-answer labels separately cached.
- Small PGM heads and ExtraTrees fit on CPU; inference stages are explicit device-masked child processes.

The official generation frameworks are pinned in `.venv-reasc`: vLLM0.11.0 / Torch2.8.0 / Transformers4.56.2. Full packages are saved in `artifacts/runs/reasc_pgm/requirements.lock.txt` and key versions in `environment.json`. Frozen v2's saved ExtraTrees is evaluated in its original `.venv` sklearn1.9.1 environment through a numeric-array subprocess bridge; the newly fitted heads use sklearn1.7.2.

Controlled reproduction adaptations, shared by every compared method:

1. The reference's `xVerify-3B-I` model alias could not be accessed. The official public Llama-based `IAAR-Shanghai/xVerify-3B-Ia` revision `67fdb1841f332753c2c9c597ea66e60e75476641` is pinned. The exact historical judge identity is unknown; historical paper accuracy is not directly replicated. Official prompt/template literals are preserved, including literal backslash-ns; greedy decoding is explicitly fixed.
2. Available Python3.10.12 differs from the source's Python3.10.16. Critical inference framework versions match.
3. Calibration IDs use isolated `random.Random(0)` before model loading, rather than relying on global RNG state after initialization.
4. Both extraction passes are retained and costed, but the overwrite pass runs per question rather than as a second whole-dataset traversal. All methods share the resulting sample bank. Batch/RNG order therefore does not reproduce historical response bytes.
5. Runtime optimization on 2026-11-02: initial serial16/TP2/eager workers were replaced by TP1 replicas with cross-question batch128 and `FULL_DECODE_ONLY` CUDA graphs, compilation level0. Completed questions and existing calibration first responses are retained byte-for-byte; per-response identities and old/new generation manifests disclose mixed execution provenance. New response/extraction requests use deterministic seed0-based hashes per question/sample/phase, while temperature1/top_p.95/top_k64/max_tokens1024/logprobs20 remain fixed. Batching and RNG streams differ from historical source bytes. The collected bank is identical across all compared methods. See `runtime_migration.json` for preservation hashes and `runtime_benchmark.json` for the dev-only speed measurement.
6. Main accuracy uses the selected candidate judged directly. The separate reference-code profile uses raw-response verdict votes and true ties; ESC retains the reference full-bank voting metric. No verdict or gold enters policy inference.

The implementation preserves the reference offline `lowest_group` calibration versus `Bottom10` inference mismatch, suffix windows, floor rounding, weighted-Beta t1 stopping, strict answer strings and ESC accumulated-prefix majority. These are not silently repaired in the baseline.

Execution order:

1. Sixteen dev questions traverse generation, both extractions, TF features, and both judge-label profiles. The audit checks actual off-argmax sampled tokens and native-logprob alignment, capacity32 versus sixteen prefixes, finite features and token/truncation counts.
2. Calibration128 → train800 → remaining dev200 are processed by concurrent generation/feature/judge workers. Completed smoke questions are reused. Automatic resume uses atomic files and stage locks.
3. Refit mixture and matched-random; evaluate the same mixture head at single partition slot1; fit global temperature and horizon4 continuation. Pick policies exclusively on dev, keeping the specified SC16−1pp feasibility criterion. Infeasibility is recorded, never relaxed using test results.
4. Save `selection_lock.json` with source/config/cohort/checkpoint/scaler/threshold/calibrator/continuation hashes before either test bank is generated.
5. Generate/evaluate MATH500500 and fresh500 using the frozen policies. Write twelve method tables, per-question decisions, corrected and reference accuracies, cost breakdowns, 10,000 paired question bootstrap intervals, and McNemar comparisons.

Main methods: pass1, SC16, ASC-code, ESC-code, ReASC-offline-code, ReASC-online-code, PGM mixture. Diagnostic methods: frozen-v2 transfer, same-head single partition, matched random, fixed95 and no-continuation mixture.

Costs are reported as reference consumed-chat proxy `4e9 * chat_tokens / 1e12`, generated/extraction/TF tokens and measured offline times, plus a separate rounded4B dense `2N` approximation. The latter includes both extraction passes and the PGM TF pass. Cached-bank collection time is not causal serving latency. Judge costs are evaluation-only. Training-seed robustness checks, OOF continuation and additional datasets remain follow-up work, outside this initial comparison.

Resume the supervisor from the repository root:

```bash
.venv-reasc/bin/python -m pgm_asc.experiments.reasc_runner
```

The supervisor owns a file lock and fails with `failure.json` if a child fails. It never terminates unrelated GPU processes. Logs are under `artifacts/runs/reasc_pgm/logs/`; initial smoke logs are in the run root. Results appear only after measured evaluation in `math500/report.md`, `fresh/report.md`, and `result_summary.md`.

Runtime benchmark: on one GPU4 B200, batch128 generated64,667 tokens in14.47seconds (4,470 output tokens/s), including native logprobs20; dev prompts only. This is a generation benchmark, not an end-to-end experiment completion estimate. The previous20-hour estimate was an extrapolation of the inefficient startup collector and is superseded after migration. Two-replica throughput and feature/judge completion must be measured before publishing an updated full ETA.
