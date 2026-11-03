# Documentation

## Current study

| Document | Contents |
|---|---|
| [Follow-up protocol](PGM_FOLLOWUP_PLAN.md) | Matched PGM ablations, model/dataset matrix, statistical analysis |
| [Completed results](../results/followup/README.md) | All completed MATH and GSM8K evaluations; fixed95 and primary contrasts |
| [Reproduction guide](REPRODUCIBILITY.md) | Environments, upstream setup, local artifact requirements, commands |
| [ReASC comparison plan](REASC_PGM_COMPARISON_PLAN.md) | Controlled reproduction and initial Gemma comparison |
| [ReASC execution notes](REASC_PGM_EXECUTION.md) | Sampling, extraction, judge, runtime adaptations |
| [Reference audit](REASC_REFERENCE_AUDIT.md) | Original archive and official implementation comparison |

The recorded protocol includes an Omni-MATH transfer experiment. That part was
subsequently deferred; the retained configs are planned cells, not completed
results. Current MATH/GSM8K learned policies have no continuation head.

## Historical V2/V5 work

These files describe the earlier integrated selector/state/value experiments.
Their settings and results should not be conflated with the current study.
Their original paths and contents are retained because historical experiment
locks hash them.

| Document | Contents |
|---|---|
| [Method](METHOD.md) | Earlier selector, temporal state and continuation models |
| [Architecture](ARCHITECTURE.md) | Package layers and legacy pipeline layout |
| [Baselines](BASELINES.md) | Earlier baseline implementation scope |
| [Experiment history](EXPERIMENT_HISTORY.md) | Previous attempts, failures and decisions |
| [Research proposal](references/research_proposal.pdf) | Original project proposal |

For the current study, start with the follow-up protocol and the root README.
