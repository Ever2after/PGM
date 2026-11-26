# P1 joint / prefix / mixed objective ablation

See the [P1 guide](P1.md) for the committed result snapshot and isolated
reproduction commands. `report` requires a completed run and its local caches.

This experiment compares joint, prefix and mixed training on
both fixed [P1 hierarchical CRP graphs](P1_HIERARCHICAL_CRP.md).
[Results](../results/p1_objectives/README.md) and the
[HTML report](../results/p1_objectives/report.html) include the full-bank study.
The follow-up [predictive-loss ablation](P1_PREDICTIVE_ABLATION.md) adds native
next-answer NLL while freezing this study's Mixed lambda and all reference
predictions. [Follow-up results](../results/p1_predictive/README.md).

## Models and losses

The q-independent P1 keeps four parameters, and the q-linked P1 keeps five.
Answer ids and validity remain the only observation inputs. The q prior, the
question-shared Normal effect on log theta, all bounded transforms, and the
native CRP posterior/continuation generator remain unchanged.

For question i and prefix t, c*it is the accepted correct answer if it has
appeared, otherwise OTHER. Training uses the exact same per-cell unique-correct
cohort as the previous P1 Joint study, including no-correct-in-bank questions.

    J = -mean_i log p(D_i,16, c*_i,16) / 16
    C_w = mean_i sum(t=1..16) w_t [-log p(c*_it | D_it)]
    L_joint = J
    L_prefix = C_w
    L_mixed = (J + lambda*C_w) / (1 + lambda)

Each data loss receives the same `6.25e-6 * sum(raw_parameter^2)` ridge.
Uniform prefix weights are 1/16 each. The second, separate weighting uses the
actual v1 band totals .4/.2/.2/.2 over t=1–2/3–4/5–8/9–16.

Mixed lambda candidates are .1/1/10, separately for each cell, graph and
weighting. The best restart is selected by penalized training loss. Lambda is
selected by unique-correct dev NLL, averaged equally over evaluation prefixes
1/2/4/8/16. Lambda=0 is the separate frozen Joint reference, rather than a mixed
candidate. Future scores and test outcomes never select lambda or a restart.

Full-batch L-BFGS uses common shared-parameter starts for seeds 0/43/44 and
sigma starts .25/.75/1.5, with the same initialization and settings as the prior
study. Pure prefix and mixed fitting always integrate the same native P1
likelihood before normalizing over hypotheses at each prefix. They add no
posterior temperature, calibration head, latent variable, or feature.

## Existing Joint and numerical controls

The previous P1 Joint fits **and prediction arrays** are frozen references.
The runner verifies bank SHA, train-ID hash, evaluation configuration, prior
code hashes, prediction-cache SHA and question ordering. New Joint metrics are
computed from those exact same arrays and checked against the previous scores.
Thus any change from Joint is not caused by a refitted reference or new MC draw.

The q grid remains at 80 logit nodes on [-7,7]. Each new objective starts at the
previous Joint cell's validated Normal quadrature resolution (61, or 121 for
Gemma/MATH). It compares the best fit at fixed parameters against the next
resolution in 61/121/181/241 on:

- training joint, weighted prefix and combined data loss;
- dev unique posterior NLL at fixed evaluation prefixes;
- maximum individual dev posterior and exact next-answer probability changes.

Every absolute difference must be <=.001. Otherwise that candidate objective
is refit at the finer resolution before any test evaluation. The chosen mixed
lambda must also have the same ranking under finer dev re-integration; an
unstable ranking stops evaluation. Full resolution histories and restarts are
saved. Integration resolution can differ between objectives solely to meet a
numerical tolerance; the probability model and q support do not change.

## Paired evaluation

The bank, labels, question order, fixed plurality reference, prefixes, horizons,
MC seeds, number of continuations and question bootstrap match the prior study.
The inference and simulation functions are imported unchanged from P1.

- Dev and test are separate. Prefixes: 1/2/4/8/16. Horizons: 1/2/4/8, t+h<=16.
- Plurality uses valid exact-string answers, breaking ties by first appearance.
- Exact next-answer distributions integrate c,q,theta; they do not use MC.
- MC uses the same question/prefix/seed inverse-CDF uniforms across all methods.
  One joint c,q,theta draw is fixed over the entire continuation.
- Three seeds x256 continuations. Gain is the same continuation's future-minus-
  current correctness, and equals repair-minus-damage at finite draw count.
- Proper posterior scores, future-event scores, repair/damage Brier/NLL/bias,
  reliability, gain MSE/bias/zero baseline/MC correction are exported.
- Question-level prefix/horizon means enter the 1,000-replicate paired bootstrap.
  Macro summaries weight nine cells equally without a pooled significance claim.
- Contrasts include each new objective minus its own graph's Joint, linked minus
  independent at each objective, and Mixed minus Prefix at each weighting.
- All-question future scores and unique-correct-only sensitivity are separate.

The evaluation retains the original xVerify labels, exact-string answer
representation, q-grid truncation and sigma bound. It does not add a generator
for empty answers. Existing test-bank exploration and model-selection
uncertainty limit generalization claims. It does not tune stopping thresholds
or measure a new token-savings operating point.

## Run and outputs

    pgm p1-objective fit --workers 4
    pgm p1-objective run --workers 4
    pgm p1-objective report

Supported overrides: `--models`, `--datasets`, `--out`, `--config`, `--overwrite`.
The default is [configs/p1_objectives.json](../configs/p1_objectives.json).
The default frozen reference is `results/hierarchical_crp`; a separate reference
folder can be supplied as `baseline_dir` in a custom configuration.

- `protocol.json`: configuration, model parameters, weights and code hashes.
- `selection/*.json`: shared cohort IDs/hash, all candidates/restarts/numerical
  histories, lambda ranking checks and the fit lock before test is loaded.
- `cells/*.json`: complete evaluation, paired question intervals and provenance.
- `summary.csv`, `macro_summary.csv`, `dataset_summary.csv`: all score summaries.
- `paired_contrasts.csv`, `paired_macro.csv`: direct contrasts, cell-level
  intervals and counts of negative differences/intervals.
- `posterior_by_prefix.csv`, `future_by_horizon.csv`, `*_calibration.csv`:
  detailed scores and reliability bins.
- `future_unique_summary.csv`, `mc_seed_sensitivity.csv`: sensitivity checks.
- `mc_seed_contrasts.csv`: per-seed, per-cell differences from the graph's Joint.
- `selection.csv`, `training_restarts.csv`, `parameters.csv`,
  `quadrature_checks.csv`, `validation.json`: selection and numerical QA.
- `artifacts/p1_objectives/<id>/`: resumable candidate fits, locked fit records
  and full compressed posterior/forecast predictions, isolated from prior runs.

Tests compare training prefix losses against native P1 posterior probabilities,
check the sigma=0 nested M1 loss and gradients, preserve shared starts and
parameter counts, reproduce Joint training, check all objective forms and
integration scores, and exercise frozen-Joint evaluation/report/resume on a
tiny bank. Existing P1 inference/sampling and M1 objective tests remain active.
