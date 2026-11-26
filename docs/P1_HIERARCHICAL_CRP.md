# P1: question-level hierarchical CRP

See the [P1 guide](P1.md) for the committed result snapshot and isolated
reproduction commands. `report` requires a completed run and its local caches.

This study adds question-to-question variation in wrong-answer concentration,
while retaining the released M1 answer representation, q prior, and observation
inputs. [Results and interpretation](../results/hierarchical_crp/README.md),
[HTML report](../results/hierarchical_crp/report.html), and
[configuration](../configs/hierarchical.json) accompany the implementation.
The follow-up [P1 objective ablation](P1_OBJECTIVE_ABLATION.md) applies prefix and
mixed learning to both P1 graphs while preserving this study's Joint predictions
as frozen references. [Follow-up results](../results/p1_objectives/README.md).

## Model and ablations

For each question i:

    q_i ~ Beta(a,b)
    u_i ~ Normal(0,1)
    log theta_i = clip(t0 + t1(2q_i-1), -8, 8) + sigma*u_i
    z_ij ~ Bernoulli(q_i)
    answer_ij = c_i if z_ij else CRP(theta_i)

The bounded t0/t1 transforms and q quadrature are identical to M1. The Normal
effect is integrated once over the **entire question sequence**, and the same
sampled theta is retained throughout each future continuation. The effect is
added after the existing deterministic base clip, so the random effect is not
censored. Small theta concentrates wrong answers on fewer answer strings.

| Variant | Base log theta | Random effect | Parameters |
|---|---|---|---:|
| M1 | t0 + t1(2q−1) | none | 4 |
| constant_theta | t0 | none | 3 |
| P1_independent | t0 | sigma*u | 4 |
| P1_linked | t0 + t1(2q−1) | sigma*u | 5 |
| quadratic_theta | t0 + t1(2q−1) + t2(2q−1)^2 | none | 5 |

Removing the q link imposes **prior** independence between q and theta.
Observation-conditioned dependence can still arise through c and the observed
answers. The quadratic control matches linked P1's parameter count. It tests
whether added deterministic q-dependence explains the benefit of random theta.

## Controlled fitting

All variants use exactly the same per-cell unique-correct training questions
and labels as the prior controlled M1 Joint experiment. Questions with no
correct answer in the bank remain in the cohort and target OTHER. Questions
with more than one accepted exact-string answer are excluded from training.
Input features remain answer ids and validity only; gold labels enter training
targets and evaluation, never posterior or forecast inference.

The loss is `−mean log p(D16,c*)/16 + (1e−4/16)*sum(raw_parameter^2)`.
All arms use full-batch L-BFGS with the same settings and common initial values
for shared parameters (seeds 0,43,44). Each arm chooses its restart by minimum
penalized training loss. No posterior or future test outcome selects a fit.

The signed Gaussian loading is `3*tanh(sigma_raw/3)`. Normal symmetry makes its
sign irrelevant; reported sigma is the absolute loading. This gives an exact
sigma=0 nested model with a ridge that favors zero. Because the derivative at
zero is zero for a variance component, sigma starts at .25/.75/1.5 for the three
restarts. The upper bound 3 is a declared computational/modeling constraint.
Boundary flags and raw estimates are exported; it is not a learned constant.

## Integration and predictive sampling

The original q grid is unchanged: 80 logit nodes over [−7,7]. Standard-Normal
Gauss-Hermite nodes integrate u. The integration ladder is 61→121→181→241.
After fitting at one resolution, the same parameters are checked against the
next resolution on train joint NLL, dev unique posterior NLL, and maximum
dev posterior/next-answer probability differences. All errors must be ≤.001.
Otherwise both random-effect arms are refit at the finer resolution. If no
resolution passes, evaluation stops before reading test. Resolution selection
is a numerical check, not a hyperparameter choice using test outcomes.

The CRP likelihood is the native sequential count likelihood. To avoid
cancellation for large theta, its rising factorial uses cumulative
`logaddexp(logtheta, log(r))` rather than a difference of two large lgamma values.
Repeated sufficient states are integrated once and mapped to each hypothesis.

Given observed answers, inference integrates the joint posterior over c,q,u.
The next-answer categorical distribution is exact under the numerical grid.
For MC continuations, one inverse-CDF uniform draws c and then its conditional
(q,u). The same question/prefix/seed uniforms are used across model variants.
Each continuation keeps c,q,theta fixed and reuses the shared native CRP
simulator. This is common-random-number coupling; it does not equate latent
states across different posterior distributions.

## Evaluation

- Existing 3 LLM × 3 dataset bank; dev and test reported separately.
- Prefixes 1/2/4/8/16 and horizons 1/2/4/8 with t+h≤16.
- Reference decision: plurality among valid exact-string answers; earliest
  observed answer wins ties. No posterior-based answer tie break.
- Unique-correct NLL/Brier, reference and OTHER calibration.
- Exact next-answer NLL/Brier; future-correct, leader-change, future-reference
  probabilities; repair/damage NLL, Brier, bias and reliability.
- Three MC seeds ×256 draws. Gain is future−current correctness on the **same**
  continuation; gain=repair−damage holds at finite draws. Gain MSE, bias,
  zero-gain baseline, MC-corrected MSE and seed sensitivity are reported.
- Each question's prefix/horizon losses are averaged before bootstrap. Paired
  question bootstrap uses 1,000 replicates. Macro averages weight the 9 cells
  equally and make no pooled significance claim.
- Full-bank future scores and unique-correct future sensitivity are both
  exported. Test-bank judge labels and prior exploratory inspection limit
  claims; a new independent holdout remains necessary.

Direct structure contrasts are linked−independent, independent−constant, and
linked−quadratic, in addition to every variant−M1. Their question-level paired
intervals and counts of improving cells are exported separately.

## Run and artifacts

    pgm hierarchical fit --workers 2
    pgm hierarchical run --workers 2
    pgm hierarchical report

`--models`, `--datasets`, `--config`, `--out`, `--overwrite` are supported.
Fits and predictions are isolated from prior studies. The experiment identity
hashes numerical/inference/evaluation code and config. Unchanged runs reuse
locked fits and predictions; mismatched protocol outputs are rejected.

- `results/hierarchical_crp/protocol.json`: settings and code hashes.
- `selection/*.json`: bank hash, common train IDs/hash, every final-resolution fit/restart,
  numerical checks, selected resolution, and timestamp before test loading.
- `cells/*.json`: all scores, question-bootstrap intervals and provenance.
- `summary.csv`, `macro_summary.csv`, `dataset_summary.csv`: aggregate scores.
- `paired_contrasts.csv`, `structure_contrasts.csv`, `structure_macro.csv`:
  paired comparisons and direct structure ablations.
- `posterior_by_prefix.csv`, `future_by_horizon.csv`, `*_calibration.csv`:
  detailed scores and reliability bins.
- `training_restarts.csv`, `parameters.csv`, `quadrature_checks.csv`,
  `baseline_reproduction.csv`, `mc_seed_sensitivity.csv`: fit and numerical QA.
- `artifacts/hierarchical_crp/<id>/`: full fits and compressed predictions,
  including cached posterior arrays and every averaged forecast field.

Tests cover nested M1 likelihood/posterior/gradients including invalid answers,
Normal moments and prior independence, label-free inference, normalized exact
next-answer distributions, joint inverse-CDF draws, shared simulator parity,
paired gain identity, integration convergence, all parameter counts, and a tiny
fit/evaluate/report/resume workflow. Existing M1 objective tests remain active.
