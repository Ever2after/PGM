# P1 Mixed + multi-step repair/damage Brier loss

See the [P1 guide](P1.md) for the committed result snapshot and isolated
reproduction commands. `report` requires a completed run and its local caches.

This changes the direct event score in the
[transition-NLL study](P1_TRANSITION_ABLATION.md) while keeping the P1 graphs,
exact event kernel, banks and paired-MC evaluation unchanged.
[Results](../results/p1_brier/README.md) and the
[HTML report](../results/p1_brier/report.html) include every fixed coefficient.

## Objective

    M = (J + lambda*C_w) / (1 + lambda)
    B = mean(i,t,h) [((r - Y_repair)^2 + (d - Y_damage)^2) / 2]
    L = (M + beta*B) / (1 + beta) + rho*sum(raw_parameter^2)

B is the mean of **two binary Brier scores**. It does not include a stay-class
square, rare-class weights, or a gain-regression term. The native r,d probabilities
are identical to the preceding study: c,q,theta are jointly marginalized given
past answers only. Gold current/future correctness enters targets, never the
forecast inputs.

The previous Mixed lambda stays fixed for each cell/graph/prefix weighting.
Beta=.1/1/10/100 are all prespecified arms. The upper range extends the sweep
because raw Brier loss and its gradients have a smaller scale than transition
NLL. No coefficient is selected using dev/test/gain scores. Equal numerical
beta values across NLL and Brier do not imply equal effective training strength.

The same normalized-loss convention and raw ridge 6.25e-6 are retained. Loss
units and the relative influence of ridge at strong beta are therefore part of
this declared setup. The experiment does not introduce a normalization learned
from the data, a new calibration head, or new model parameters.

## Fixed controls

- Both q-independent/linked P1 graphs retain 4/5 parameters and answer-id/valid
  inputs, all priors/bounds and the original q grid.
- The same unique-correct train question IDs, including no-correct-in-bank
  questions, are used as in Mixed and transition NLL.
- Prefixes 1/2/4/8 and horizons 1/2/4/8 with t+h<=16 enter B uniformly within
  each question. Uniform/v1 posterior-prefix weighting are separate blocks.
- Full-batch L-BFGS, starts 0/43/44, sigma starts .25/.75/1.5 and optimizer
  settings are unchanged. The optimizer implementation is shared with NLL;
  the event-score component is replaced and its metadata renamed to Brier.
- Beta=0 reproduces the Mixed data loss and parameter initialization.
- Previous Mixed and all transition-NLL coefficient arms are frozen references.
  The fixed paired NLL comparator is beta=1; it is not selected from test scores.

The exact finite-horizon CRP path kernel is imported unchanged from the NLL
study. That kernel has been checked against full path enumeration, an
independent exact h=8 dynamic program, native simulation and latent quadrature.
No discrete MC sample is differentiated during training.

New tests verify B against the independently scored repair/damage probabilities,
unchanged P1 event probabilities, gold-label isolation, finite-difference
Brier gradients, the normalized mixed objective and beta=0 behavior. The full
frozen-reference fit/evaluate/report/resume workflow is also tested.

## Evaluation

Native evaluation preserves the original three seeds x256 paired continuations,
question/prefix/seed uniforms, fixed plurality decision and first-appearance tie
rule. Each c,q,theta draw is held throughout the continuation, so finite-draw
gain=repair−damage remains exact.

Frozen native **and exact-event** reference arrays are reused. A separate exact
event sensitivity for the new models removes R/D/gain MC noise using the same
kernel and numerical quadrature. Bank/cohort/code hashes, cache SHA, question
order, frozen arrays and common summary scores are checked. Fit locks precede
test loading.

Scores cover posterior quality, exact next-answer prediction, multi-step events,
individual and mean R/D Brier, transition log score, repair/damage reliability,
gain MSE/bias/zero baseline/MC correction, MC-seed sensitivity and unique-only
future sensitivity. Paired question intervals compare each Brier arm with its
Mixed and fixed NLL-beta=1 references, and compare the two graphs.
The report also macro-averages the existing fixed-bin, prefix/horizon gain ECE.
It is a descriptive reliability statistic, not an independent calibration test.

Question prefix/horizon means enter the 1,000-replicate paired bootstrap. Macro
results weight cells equally, without a pooled significance claim.

Every new fit is checked against finer Normal quadrature on train J/C/B/data
loss, dev posterior/Brier and individual dev posterior/next/event probabilities.
Any absolute error above .001 triggers refitting before test. All histories
are saved; neither q support nor the probability model changes.

## Gain-error decomposition

For `e_R=r−Y_R` and `e_D=d−Y_D`,

    Gain MSE = Repair Brier + Damage Brier - 2*mean(e_R*e_D).

The training loss B is half the sum of the first two terms. Within a restricted
PGM family, a reduction in this sum does not by itself guarantee a reduction in
gain MSE, because the cross term can change. The empirical identity is checked
using directly calculated errors, rather than defining the cross term from
MSE. Per-cell and macro terms are exported for native MC and exact evaluation.

Posterior retention uses `(Joint NLL - new NLL)/(Joint NLL - Mixed NLL)` where
the denominator is positive. Joint is read from the verified historical study;
no Joint fit is retuned. Macro and per-cell retention are descriptive outcomes,
not coefficient-selection rules.

## Scope and reproducibility

This is an exploratory comparison on the already inspected xVerify exact-string
bank. Multi-correct strings, q truncation, bounded priors/sigma, finite sample
uncertainty and the missing invalid-answer generator remain limitations.
Results apply to the declared fixed lambda/beta grid and decision/horizon
weights. The experiment does not compare one-step-only transition training,
gain-only regression or actual stopping/token-savings operating points.

    pgm p1-brier fit --workers 4
    pgm p1-brier run --workers 4
    pgm p1-brier report

Settings: [configs/p1_brier.json](../configs/p1_brier.json). Standard overrides
are `--models`, `--datasets`, `--out`, `--config`, `--overwrite`.
`baseline_dir` can identify a compatible transition-NLL run; the default is
`results/p1_transitions`.

`results/p1_brier/` contains protocol/selection locks, per-cell JSON, native and
exact summary/paired tables, posterior retention, gain-error decomposition,
calibration/prefix/horizon tables, unique/MC sensitivity, restart/parameter and
quadrature diagnostics, and validation. Large arrays and resumable candidates
are stored separately under `artifacts/p1_brier/<experiment_id>/`.
