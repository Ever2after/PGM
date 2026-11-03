# Baseline reproduction notes

All methods consume the same generated response bank within each experimental stage. Generated-token cost excludes teacher-forcing features and head fitting. Gold is used only for supervised fitting, calibration/policy selection, and evaluation.

| Method | Source / scope | Retained implementation |
|---|---|---|
| SC | Self-Consistency, ICLR2023 | Fixed-budget majority vote, first-observed tie breaking |
| ASC | Adaptive Self-Consistency, EMNLP2023 | Beta top-two confidence; stopping threshold selected on calibration-B |
| ReASC | arXiv2601.02970v1 implementation target | Full-vocabulary self-certainty, bottom10% of stride1/window128 means; first-response normalization/gate on calibration-A; weighted counts and ASC confidence |
| CGES | arXiv2511.02603v2 implementation target | Geometric LNS confidence; likelihood recomputed for all previous responses at the current answer-support size, with OTHER |

Source links: [SC](https://arxiv.org/abs/2203.11171), [ASC](https://aclanthology.org/2023.emnlp-main.761/), [ReASC implementation version](https://arxiv.org/abs/2601.02970v1), [CGES implementation version](https://arxiv.org/abs/2511.02603v2). Later publication versions have not been silently substituted for these pinned implementations.

CGES confidence definition was originally selected on selector dev200 and is now fixed in the unified config. In the state stage, global temperature is fitted on fresh A150 and stopping threshold on B300. ReASC uses only the first response per A question for confidence normalization/gate:150 response examples. All state-stage adaptive methods share minimum2/cap8; SC uses fixed2/4/8.

In the selector stage, matched continuation learners are added to ASC/CGES using train800/dev200 and selected on B300. ReASC retains the earlier policy because the old training bank does not contain the required full-vocabulary confidence. These enhanced selector-stage controls are not the native state-stage baselines.

Shared functions are in `pgm_asc/policies/baselines.py`; supplemental full-vocabulary extraction is in `pgm_asc/collection/confidence.py`. Original reproduction notes and ESC's earlier experiment remain as text in EXPERIMENT_HISTORY.md. ESC is not a state-stage comparison.
