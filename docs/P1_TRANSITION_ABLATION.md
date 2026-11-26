# P1 Mixed + direct multi-step repair/damage loss

See the [P1 guide](P1.md) for the committed result snapshot and isolated
reproduction commands. `report` requires a completed run and its local caches.

This study adds a proper finite-horizon transition score to the fixed P1 Mixed
objective. It compares against the frozen Mixed and
[next-answer-loss models](P1_PREDICTIVE_ABLATION.md), using the same bank,
question cohorts and paired Monte Carlo protocol.
[Results](../results/p1_transitions/README.md) and
[HTML report](../results/p1_transitions/report.html) include all coefficient arms.
The follow-up [Brier ablation](P1_BRIER_ABLATION.md) changes the direct event
score to mean binary repair/damage Brier while freezing this study's references.
[Follow-up results](../results/p1_brier/README.md).

## Objective and controls

    M = (J + lambda*C_w) / (1 + lambda)
    p_transition(D_t,h) = (r, d, 1-r-d)
    T = mean_i mean_(t,h) [-log p_transition(actual transition)]
    L = (M + beta*T) / (1 + beta) + rho*sum(raw_parameter^2)

The classes are repair (incorrect→correct), damage (correct→incorrect), and stay
(no change in correctness, including wrong→another wrong answer). They are
mutually exclusive. This is an unweighted proper categorical log score; there
is no rare-class reweighting or direct gain-only regression.

Transition probabilities marginalize c,q,theta given **past answers only**.
The current correctness label and the gold correct-answer hypothesis are not
provided to the forecast. Observed current/future correctness enters only the
training target and evaluation.

The previous per-cell, per-graph, per-prefix-weighting Mixed lambda is frozen.
The prespecified beta arms .1/1/10 are all reported; beta=0 is the previous
Mixed reference. No dev/test score selects beta or retunes lambda. The
next-answer comparator for paired contrasts is fixed at gamma=1, and all
previous next-answer gamma arms remain in the saved tables.

Both P1 graphs retain 4/5 parameters, answer-id/valid inputs, q prior/grid,
question-shared lognormal theta effect and all bounds. The exact same common
unique-correct training questions are used, including no-correct-in-bank
questions. Full-batch L-BFGS, seeds 0/43/44, sigma starts .25/.75/1.5 and raw
ridge 6.25e-6 remain unchanged. Restart selection uses penalized training loss.

T uses every feasible pair from prefixes 1/2/4/8 and horizons 1/2/4/8, with
uniform pair weights within a question. This matches the prior evaluation
pairs. Uniform/v1 are separate posterior-prefix weighting arms; T has the same
pair weighting in both.

## Exact native transition probabilities

Conditional on a fixed c,q,theta, sum over k correct additions and w=h−k wrong
additions. The correct-addition factor is `choose(h,k)*q^k*(1-q)^w`. The wrong
continuation is the original CRP, retaining the same theta throughout.

Let the correct answer's final count be L. If c is already observed, each wrong
answer first seen before c must finish with count <=L−1; later wrong answers
and future new wrong answers may finish with count <=L. If c is unseen, every
existing wrong answer must finish below L. For future new wrong groups tying
an unseen c at count L, c appears first among r+1 equal-sized groups with
probability 1/(r+1).

For an old wrong group of size n with cap B, its increment exponential
generating function is `sum_r (n)_r*x^r/r!`, truncated at B−n. New wrong groups
use the CRP cycle weights in `exp(theta*sum_(s<=L) x^s/s)`. The unseen-correct
case additionally weights the number r of new groups of size L by 1/(r+1).
Multiply the old/new generating functions and extract the w-th coefficient.
Each resulting term has weight `theta^j/(theta+m)_w` for j new wrong groups.

Multiplying by the observed-prefix likelihood gives the same P1 sufficient
state with increment `(k,w,w,j)` in `(correct count, wrong Bernoulli count,
valid wrong count, distinct wrong groups)`. These augmented states are
integrated with the existing q/Normal quadrature. This sums all continuation
paths without differentiating discrete MC samples. Prefix contexts with the
same count vector and first-appearance ordering share computation.

For each c hypothesis, F_c is its posterior-weighted probability of being the
future plurality answer. Then `repair=sum_(c != current leader) F_c` and
`damage=p(current leader correct)-F_current_leader`. Impossible damage events
are set to zero analytically. The native fixed plurality rule and first-valid-
appearance tie order are preserved.

Validation includes full labelled path enumeration through h=3, an independent
finite-state dynamic program through h=8, high-draw native simulation at h=8,
posterior×conditional-kernel quadrature, finite-difference gradients, normalized
R/D/stay probabilities and label/suffix isolation. Beta=0 reproduces Mixed.
The horizon-eight independent DP does not use the generating-function weights.

## Paired evaluation and exact sensitivity

All prior parameter/prediction arrays are frozen, with bank/cohort/code hashes,
prediction-cache SHA and question order checked. Fit locks precede test loading.
The same 3 seeds x256 native continuations use common question/prefix/seed
uniforms across models. Each continuation draws c,q,theta jointly once and
retains them. Finite-draw gain=repair−damage remains exact.

The main table uses the unchanged paired MC. A separate exact-event sensitivity
recomputes R/D, gain, future reference correctness and the three-class log score
from the same model using the exact path sum. It therefore removes event-MC
noise while retaining numerical q/Normal integration. It does not change the
model or substitute a different reference decision.

Each new arm is compared against its own Mixed and fixed next-answer gamma=1
model, and q-linked/independent arms are paired at each beta. Question bootstrap
uses 1,000 replicates after averaging a question's prefix/horizon scores. Macro
averages weight cells equally without a pooled significance claim. Full-bank
scores and unique-correct future sensitivity are provided.

Each candidate is checked at fixed parameters against finer Normal quadrature
on train J/C/T/data loss, dev posterior/transition NLL, and maximum dev
posterior/next-answer/event probability changes. Errors above .001 trigger
refitting before test; all resolution histories are saved. q support and the
model's prior bounds remain unchanged.

## Scope

These are exploratory comparisons on the existing xVerify exact-string bank.
They do not independently validate a final stopping policy or token savings.
The finite-horizon loss targets the fixed decision rule and horizon grid;
generalization to different horizons or adaptive stopping-state frequencies is
not established. Judge uncertainty, multi-correct strings, q truncation, bounds
and the missing invalid-answer generator remain limitations.

This is specifically a **categorical transition-NLL** ablation. It does not
compare one-step transition-only training against the multi-step version, or
compare transition Brier training. Thus it does not isolate target changes from
horizon changes, and a negative result does not rule out every direct event
loss or establish a PGM expressiveness limit.

## Reproduce and inspect

    pgm p1-transition fit --workers 4
    pgm p1-transition run --workers 4
    pgm p1-transition report

Settings: [configs/p1_transitions.json](../configs/p1_transitions.json).
Supported overrides are `--models`, `--datasets`, `--out`, `--config`,
`--overwrite`. `baseline_dir` can point at a compatible previous predictive
study; the default is `results/p1_predictive`.

Outputs in `results/p1_transitions/` include:

- protocol and selection locks; every candidate/restart/numerical history;
- native summary, macro/dataset, prefix/horizon and calibration tables;
- paired contrasts against Mixed/next-answer and between the two graphs;
- exact transition summaries, calibration and paired contrasts;
- macro posterior retention and per-cell retention/contrast diagnostics;
- MC seed sensitivity/contrasts, unique-only future sensitivity and validation.

Large native/exact predictions and resumable fits live under
`artifacts/p1_transitions/<experiment_id>/`. Existing study outputs are preserved.
