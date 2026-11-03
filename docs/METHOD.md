# Integrated PGM adaptive self-consistency

Historical V2 and V5 are now stages of one pipeline: **response bank → positive-support answer selector → optional temporal state/value model → calibration → locked evaluation**. This is a code and artifact reorganization. Existing model weights, cohort membership, policies, and numerical results are preserved; no new experiment was run during the refactor.

## Shared generation and features

- Frozen `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`; no LLM fine-tuning.
- `EleutherAI/hendrycks_math`, revision `21a5633873b6a120296cce3e2df9d5550074f4a3`.
- vLLM sampling: temperature 0.7, top-p 0.95, max new tokens4096, context8192, seed42; deterministic question/sample seeds.
- Teacher-forcing on the exact generated token IDs extracts confidence and final-layer reasoning embedding. ReASC uses separate full-vocabulary confidence. Gold labels are excluded from generation and inference inputs.
- Feature preprocessing is fitted on the selector train800. Partition thresholds use 102,400 train-only sampled embedding pairs. Scalers and representation are shared by both stages.

## Stage 1: positive-support selector and stopping policy (historical V2)

For response i, the support MLP maps nine observed features to a positive evidence weight. Six partition candidates comprise all-one, four cosine thresholds, and singletons, with duplicate partitions masked. Cluster evidence is discounted using learned alpha; a learned gate mixes their answer distributions. An OTHER head represents gold answers not yet observed. All these heads are jointly trained on the same 800 questions.

| Module | Training / fitting data | Size | Selection data |
|---|---|---:|---|
| Support / gate / OTHER / alpha | MATH train800, 32 responses each |25,600 prefix histories, t1–32|MATH train dev200, 6,400 histories|
| Prefix-balanced refit | Same train800 |Six equal-mass prefix buckets:1,2,3–4,5–8,9–16,17–32|Same dev200|
| Global temperature | Fresh MATH-test calibration-A150 |4,800 prefix histories|—|
| Minimum / patience / cap / gamma | No head fitting |Policy comparison on calibration-B300|Same B300|
| ExtraTrees continuation | Same train800, t1–7 |5,600 histories|Same dev200, 1,400 histories; lambda on B300|
| Locked evaluation | Separate MATH-test holdout500 |16,000-response bank|Evaluation only|

The retained stage ladder is S0 global → S1 minimum → S2 patience → S3 cap → S4 prefix calibration → S5 prefix loss → S6 continuation. A/B cohorts are reused across this ladder. The final mixture uses minimum2, patience1, cap8, global temperature, prefix-balanced heads, and horizon4 continuation. Candidate changes need calibration cost improvement under the empirical accuracy constraint; not every stage is adopted.

ASC/CGES receive matched continuation controls in this stage; ReASC retains its earlier policy because the old training bank lacks its full-vocabulary confidence. These enhanced controls are distinct from the native baselines in stage2.

## Stage 2: frozen selector, temporal state, and continuation value (historical V5)

Selector M uses the stage1 mixture. S fixes partition slot1 while keeping the same support/OTHER/alpha. A and I fix all-one/singleton partitions; N is the separately trained native single-partition checkpoint. Selector weights remain frozen.

State U means the gold answer is unseen; W means seen but the selected answer is wrong; R means the selected answer is correct. MLP, unmasked chain, and masked chain share 43 observed features. Transition masks encode when a new answer or selector change makes a state transition possible. Inference is causal.

| Module / role | Cohort | Size |
|---|---|---:|
| State/value train | Spent V2 A150+B300 and V3 A150+B300 |900 questions, first8 responses each;7,200 packed histories|
| State direct NLL | Same train |6,300 nonzero-weight histories, t2–8|
| State OOF | Question-level5-fold |720 train /180 held questions per fold; deployment uses full900 fit|
| Fixed4 Ridge | Same train, t2–7 |5,400 rows, predicting correctness gain and added token cost|
| Bellman Ridge | Same train, six decision times |900 rows per decision time per lambda|
| Dev | Old selector dev200 + spent selector holdout500 |700 questions, first8 responses each|
| Temperature calibration | New A150 |1,050 direct-loss prefixes, t2–8|
| Threshold/lambda/family selection | New B300 |300 question-level decisions|
| Locked holdout | New1,000 |8,000 responses, evaluation only|

Twelve arms use72 state-model fits including folds/full fits and2,232 Ridge fits including outer folds. Outer-fold held-question labels are excluded from the new state/scaler/value fitting. The frozen selector's historical policy selection did use450 of these900 training questions; this historical dependency is not covered by the new OOF claim. Inner value training still uses fitted state predictions.

Native SC/ASC/CGES/ReASC do not train a neural head on the900 questions. ASC selects its threshold on B300. CGES retains its historical confidence definition and fits temperature on A150. ReASC calibrates normalization/gate from the first response of each A question:150 response examples. All adaptive policies use minimum2 and cap8. Fresh responses total11,600.

## Canonical inputs and provenance

`artifacts/runs/pgm_asc/bootstrap/cache` stores the original selector training tensors, representation and scalers. `state/cache/train` and `state/cache/dev` are canonical packed inputs for the state/value training. Their question text and separate gold files are under `artifacts/data/pgm_asc/selector_train` and `state_train`.

V1/V3 raw generation banks and discarded models were deleted. Rebuilding their packed inputs from those raw files is intentionally unavailable; training from the retained packed inputs remains supported. V2/V5 calibration and evaluation response banks remain available. Historical question IDs are retained to preserve sample identity and random seeds.

`excluded_problem_hashes.json` preserves normalized identities from every previous cohort, including deleted experiments, to avoid resampling spent questions. Migrated integrity locks record current paths/hashes after evaluation; their original pre-holdout locking history is documented in EXPERIMENT_HISTORY.md. A migration lock is not a new preregistration.

## Interpretation

Selector holdout500 and state holdout1000 differ. Compare methods within a cohort. The state-stage primary reduces token cost but loses too much accuracy; it is not promoted over the frozen selector. Cost is generated tokens, not end-to-end latency, and repeated development/calibration choices are exploratory rather than a population risk guarantee.
