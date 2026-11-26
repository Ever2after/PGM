# P1 Mixed + next-answer predictive NLL

See the [P1 guide](P1.md) for the committed result snapshot and isolated
reproduction commands. `report` requires a completed run and its local caches.

This extends the [P1 objective ablation](P1_OBJECTIVE_ABLATION.md) with a native
one-step answer prediction loss. [Results](../results/p1_predictive/README.md)
and the [HTML report](../results/p1_predictive/report.html) retain all fixed
coefficient arms, both q-linked/independent graphs, and both prefix weightings.
The follow-up [direct-transition study](P1_TRANSITION_ABLATION.md) adds a native
multi-step repair/damage/stay loss to Mixed and preserves these predictions as
frozen references. [Follow-up results](../results/p1_transitions/README.md).

## Controlled objective

    J = -mean log p(D16,c*) / 16
    C_w = mean_i sum(t=1..16) w_t [-log p(c*_it | D_it)]
    M = (J + lambda*C_w) / (1 + lambda)
    N = mean(i,t=1..15) [-log p(answer_i,t+1 | D_it)]
    L = (M + gamma*N) / (1 + gamma) + rho*sum(raw_parameter^2)

The previous Mixed lambda is **frozen** per cell, graph and prefix weighting.
Gamma is the relative weight on the already normalized Mixed objective. The
prespecified gamma arms are .1/1/10; gamma=0 is the frozen previous Mixed
reference. Every arm is reported, without test-based coefficient selection or
an arbitrary posterior-retention constraint.

Uniform and actual v1 prefix weights remain separate. N weights all 15
available one-step prefixes equally; evaluation retains the original prefixes
1/2/4/8/16 (next-answer scoring excludes 16). The two P1 graphs keep 4/5
parameters, original inputs/priors/bounds, and a single question-shared theta.
No predictive head, calibration temperature, feature, or observation variable
is introduced.

The exact same unique-correct training question IDs are used for J, C and N,
even though N itself needs no correctness labels. This keeps the cohort fixed.
L-BFGS, raw-parameter ridge 6.25e-6, common starts 0/43/44, and sigma starts
.25/.75/1.5 match the prior studies. Restarts are selected by penalized training
loss, independently for each arm.

## Native predictive distribution and gradients

The target distribution is `p(next exact-string answer | observed prefix)`,
with c,q,theta jointly integrated under the model posterior. Correctness labels
are not inputs to this probability. The next target is an existing candidate,
NEW, or INVALID. Both unseen-correct and new-wrong branches contribute to NEW.
A known-gold-c conditional likelihood is a different objective and is not used.

For an active correct-answer hypothesis, let its sufficient state be
`(n_c, wrong_Bernoulli_count, m, K)`, with m valid wrong responses and K distinct
wrong answers. The native continuation factors are q, `(1-q)/(m+theta)`, and
`(1-q)*theta/(m+theta)` for correct/repeated-wrong/new-wrong answers. Their
unnormalized integrated moments are computed by the following state increments:

| Factor | State increment |
|---|---|
| q | (1,0,0,0) |
| (1−q)/(m+theta) | (0,1,1,0) |
| (1−q)*theta/(m+theta) | (0,1,1,1) |

Repeated-wrong contributions additionally multiply by the target's existing
count. Appropriate moments are summed over c and divided by the prefix's
native hypothesis normalizer. All calculations use log-sum-exp and the same
Normal/q quadrature as P1. Shared augmented sufficient states avoid allocating
question x prefix x candidate x q x Normal tensors.

This equals the existing `Inference.next_answer` generator. For valid next
answers it also agrees with consecutive marginal-evidence ratios when the
OTHER-to-new-candidate branch is handled consistently, including newly observed
candidates. This supplies an independent numerical cross-check. For invalid
answers the legacy likelihood's evidence ratio differs from the generator,
which has no invalid-answer branch; training follows the generator.

Invalid next answers retain native probability zero. The unchanged evaluator's
1e-12 clipping supplies a constant, zero-gradient term. The study does not
silently filter these observations or add an invalid-answer model.

## Frozen references and numerical checks

Every previous Joint/Prefix/Mixed fit and full posterior/forecast array is
reused. Bank hash, training cohort hash, prior numerical code hashes, evaluation
configuration, prediction-cache SHA and question order are verified. Recomputed
reference summary scores must agree with the prior report. Prior study outputs
are not modified.

Each new arm starts at its source Mixed fit's validated Normal quadrature
resolution, keeping the original 80-node q grid. The fixed-parameter fit is
checked against the next Normal resolution in 61/121/181/241 on train J/C/N/data
losses, dev unique posterior/next NLL, and maximum dev posterior/next-answer
probability differences. Every absolute error must be <=.001; otherwise that
arm is refit before test. Full histories are saved.

Fit locks are written before test arrays are loaded. No gamma or new lambda
selection uses dev/test/gain scores; dev is used for numerical checks and
recorded diagnostics. Lambda selection belongs to the previous frozen study.

## Paired evaluation and interpretation

The same banks, original xVerify labels, fixed plurality reference, evaluation
prefixes/horizons, three MC seeds x256 continuations, and 1,000-replicate paired
question bootstrap are reused. The unchanged native P1 simulator uses common
question/prefix/seed inverse-CDF uniforms and holds each c,q,theta draw over the
entire continuation. Finite-draw gain=repair−damage remains exact.

Scores include posterior proper scores, exact next-answer scores, multi-step
future events, repair/damage scores and reliability, gain MSE/bias/zero baseline,
MC correction, seed sensitivity and unique-correct-only future sensitivity.
Contrasts include every gamma versus its own Joint/Prefix/Mixed references and
linked minus independent at every gamma/weighting.

Posterior retention is `(Joint NLL - new NLL)/(Joint NLL - Mixed NLL)` when the
denominator is positive. Macro retention uses macro NLLs; the per-cell fraction
is separately exported. Values over 100% and below zero are retained as-is.
The ratio is a descriptive outcome, never a selection rule.

Simultaneous improvement counts require lower unique posterior NLL, exact
next-answer NLL, and gain MSE than the reference on the same cell. Signed gain
bias alone is not a success criterion. All gamma results and cell-level paired
intervals remain available, without a pooled macro significance claim.

These are exploratory results on an already inspected bank. One-step answer
prediction does not guarantee better multi-step repair/damage or stopping.
Raw-string labels, q truncation, bounded priors/sigma, finite bank/MC uncertainty
and the missing invalid-answer generator remain limitations. Actual stopping
thresholds and token savings are not optimized here.

## Reproduce and inspect

    pgm p1-predictive fit --workers 4
    pgm p1-predictive run --workers 4
    pgm p1-predictive report

Settings: [configs/p1_predictive.json](../configs/p1_predictive.json).
Standard `--models`, `--datasets`, `--out`, `--config`, `--overwrite` overrides
are supported. Default references are `results/p1_objectives`; a custom
`baseline_dir` can point at a compatible prior run.

Results live in `results/p1_predictive/`: protocol and fit locks, per-cell JSON,
summary/macro/dataset tables, paired contrasts, posterior retention, prefix and
horizon scores, reliability bins, unique-only/MC sensitivity, fit/restart and
quadrature tables, parameter/bound diagnostics, and `validation.json`.
`artifacts/p1_predictive/<id>/` holds resumable candidates, all fit histories and
large compressed predictions, isolated from previous experiments.

Tests compare every predictive target probability to native P1 at all 15
prefixes (including existing/NEW/invalid and OTHER), verify finite-difference
gradients, verify label/suffix isolation, check gamma=0 and old J/C gradients,
and run the three-stage fit/evaluate/report/resume workflow on a tiny bank.
