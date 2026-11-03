# PGM necessity and model/dataset generalization study

Protocol fixed 2026-11-02, after the initial Gemma/ReASC comparison. GPU4–7 only. All new outputs live under `artifacts/{data,runs}/pgm_followup`; earlier locks, code and results remain intact.

The research question is whether semantic partition structure contributes beyond equal-budget supervised confidence aggregation, and whether the result transfers across model families, model size and reasoning difficulty. The failed continuation policy is excluded from every learned arm in this study. Mixture need not win: negative structure contrasts are reported as evidence against its necessity.

## Matrix and data

Models: pinned Gemma-3-4B-IT, Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct. This covers three model sizes and two families. Native pinned model sampling defaults are preserved: Gemma temperature1/top_p.95/top_k64/repetition1; Qwen temperature.7/top_p.8/top_k20/repetition1.05. All methods within a cell share exact samples, order, extraction and labels. Seed0-based deterministic per-request RNG, native logprobs20, sixteen responses/question, maximum1024 tokens for MATH/GSM8K and2048 for Omni-MATH. Two extraction passes are retained for compatibility and costed.

| Dataset | Fitting / selection / calibration | Evaluation |
|---|---|---|
| MATH | Same original-train800 / original-train200 dev / calibration128 as preceding study | Standard OpenAI MATH500; new confirm500 never used in any earlier registered experiment; Gemma also replays previously observed fresh500, explicitly exploratory |
| GSM8K | Official train: disjoint800 train,200 dev,128 calibration | Entire official test1,319 rows; no test labels for fitting |
| Omni-MATH | No Omni supervision or threshold tuning; transfer this model's MATH heads, scalers, thresholds, temperatures and policies | Reference archive1,000 rows, deduplicated; remove exact normalized overlap with fitting problems and report all exclusions |

`artifacts/data/pgm_followup/manifest.json` records actual counts, hashes and exclusion checks. New MATH confirm500 is uniformly sampled, seed20261103, after excluding all registered historical questions plus every cohort from the preceding comparison. GSM train/test textual overlaps and fitting duplicates are removed from the training candidates. Omni is a zero-shot cross-dataset transfer result, not an in-domain Omni refit result. Old Gemma MATH banks are linked read-only and skipped by all collection workers.

## Structure controls

Every independently fitted arm gets the same800/200/128 questions, AdamW lr.001/wd.001, hidden width32, five equal-mass prefix buckets, maximum120 epochs, dev-NLL early stopping patience20, question batch32, clip1, and seeds42/43/44. No best-seed selection or seed ensembling. Parameter counts and first-step active-gradient counts are reported; raw-embedding DeepSets has a larger encoder, so parameter equality is not claimed.

| Arm | Purpose |
|---|---|
| Mixture | Proposed positive-support partition mixture, learned gate and alpha |
| Uniform mixture | Same semantic partitions; remove learned gate |
| Matched random | Causal random grouping with identical cluster-size multiset at every prefix |
| Alpha1 | Disable cluster-size discount; response-confidence aggregation remains |
| Single partition | Independently train fixed partition slot1, threshold=train similarity10th percentile |
| All-one | All responses share one group, independently trained |
| Singletons | Every response its own group, independently trained |
| Candidate count MLP | Learned vote aggregation/OTHER using count features only |
| Candidate confidence MLP | Counts plus candidate mean confidence/length, no semantic grouping |
| DeepSets confidence | Response-confidence multiset encoder with local/global pooling |
| DeepSets embedding | Same local/global pooling with raw normalized reasoning embeddings; no explicit partitions |
| Same-head single | Copy each mixture checkpoint and fix partition slot1, isolating intervention from retraining |

Embedding DeepSets encodes responses independently, masks unobserved responses before candidate/global pooling, and is tested for future-input invariance. Capacity32 and actual prefix length16 are distinct throughout. No policy receives gold, judge labels or future aggregates.

## Operating points and reference methods

All learned arms use global temperature fit on calibration128, minimum2/patience1/cap16, no continuation. Primary operating point: choose gamma from .5/.6/.7/.8/.85/.9/.925/.95/.975/.99/.995/disabled using dev only; among candidates at least SC16 dev accuracy minus1pp, minimize total output tokens including both extraction passes. Infeasibility is recorded, never repaired using test. Secondary operating point: fixed gamma.95 for every calibrated arm. Report fixed-prefix accuracy at2/4/8/16 to isolate aggregation from stopping. This is a newly specified study; changing from the earlier chat-token selection objective is disclosed and not portrayed as the original frozen policy.

Baselines: pass1, SC16, native ASC, native ESC, native ReASC-offline and native ReASC-online. Add dev-tuned ASC/ReASC Beta thresholds using the same gamma grid, objective and SC16−1pp constraint; ReASC's calibration gate remains active. Original source semantics, including min-score calibration/Bottom10 inference mismatch and possible t1 stop, are preserved. Online GMM uses the evaluation cohort's first responses without labels; it has a cohort first-pass barrier. On Omni, native-tuned thresholds and offline calibration also transfer from MATH without Omni labels.

## Locks, evaluation and statistical claims

Each model/dataset cell locks heads, temperatures, selected thresholds, preprocessing, cohorts, config and code before generating/evaluating new test responses. Omni locks a transfer from the already frozen MATH cell. Initial Gemma MATH500 and previous fresh are development-observed benchmarks, not new confirmatory evidence. The never-used confirm500 and new GSM8K/model cells are distinguished accordingly.

Primary seed42. Seeds43/44 report stability, mean/std, never increase the effective question count. Primary structure contrasts at the dev-selected operating point: mixture vs alpha1, matched random, confidence MLP and embedding DeepSets. Report question-paired bootstrap10,000 CIs and exact McNemar, Holm-adjusting the four structure contrasts per dataset/model. Fixed95, extra structural arms, budget curves and baseline contrasts are predefined secondary analyses; report raw uncertainty and their secondary status. Do not claim global familywise significance across the whole nine-cell matrix from per-cell Holm values.

Primary label profile: same pinned official xVerify-3B-Ia candidate-level labels, greedy; secondary raw-response verdict vote profile separately reported. The original ReASC judge alias is unavailable, so this remains a controlled reproduction. Judge prompt literals are preserved; maximum judge context16,384 is an operational allowance for Omni, not a model change. Invalid verdicts/context overflow fail explicitly; no problem is silently dropped for poor outputs. Training/calibration labels never reach policy inference.

Report accuracy, consumed responses, body output tokens, output including extraction, extraction input tokens, required TF pass tokens/time, reference chat proxy and separate rounded-N dense2N approximation. The count-only control needs no TF features at serving; other current confidence/embedding arms do. Offline bank collection and head replay time are not causal serving latency. All fitting, label and benchmark collection burdens are disclosed. End-to-end serving optimization is a separate follow-up, not claimed solved here.

## Execution

CPU fitting runs independently while GPU4/5 collect with TP1 replicas, CUDA decode graphs, batch128 (Omni64). GPU6 extracts exact-token full-vocabulary confidence/final-layer reasoning embeddings with the corresponding frozen generator; GPU7 runs common xVerify. Qwen/GSM smoke validation must precede full new-cell evaluation. Collections are resumable with atomic question commits and per-shard locks; reused Gemma source directories are never written.

`configs/pgm_followup.json` and per-cell configs enumerate the queue. A persistent supervisor collects fitting cohorts, lets the CPU worker fit/lock ready cells, and evaluates locked cells while fitting subsequent cells. Live status and per-cell reports are saved in the study run root. Existing experiment artifacts and source files registered in their locks are not modified.
