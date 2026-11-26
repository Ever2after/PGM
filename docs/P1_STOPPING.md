# P1 adaptive self-consistency stopping

See the [P1 guide](P1.md) for the committed result snapshot and isolated
reproduction commands. `report` requires a completed run and its local caches.

The earlier P1 studies measured posterior and future predictions at fixed
prefixes. This experiment reuses their frozen heads and adds the original v2
stopping rule. It follows [the repository evaluation protocol](PROTOCOL.md),
including its standard plugin, scorer, report and plotting interfaces.

## Prespecified comparisons

The q-independent (4-parameter) and q-linked (5-parameter) P1 models each retain
their previous Joint, Mixed-uniform and Mixed-v1 head from
`results/p1_objectives/selection/`. The old dev-selected training lambda stays
fixed. There is no new training, no selection among these heads by stopping
results, and no change to their answer-id/valid inputs or priors.

All six heads use VOI stopping. The two Mixed-uniform heads also use confidence
stopping as a criterion ablation. Their registered names are:

| Head | VOI method | Confidence control |
|---|---|---|
| Independent Joint | `p1_independent_joint` | — |
| Linked Joint | `p1_linked_joint` | — |
| Independent Mixed uniform | `p1_independent_mixed` | `p1_independent_mixed_confidence` |
| Linked Mixed uniform | `p1_linked_mixed` | `p1_linked_mixed_confidence` |
| Independent Mixed v1 | `p1_independent_mixed_v1` | — |
| Linked Mixed v1 | `p1_linked_mixed_v1` | — |

Mixed-uniform and Joint are the primary report block; Mixed-v1 and confidence
are reported as separate sensitivity blocks. The fixed M1, SC@k, ASC, ESC and
ReASC results remain the baseline references.

## Stopping statistic

At every prefix t, A(t) is the posterior probability of the answer selected by
the original v2 plurality decision, with posterior tie-breaking. The generator
draws (c,q,theta) jointly once per continuation and holds it fixed throughout.
At each future horizon, the model updates its posterior with the simulated
answers and computes the confidence of the future plurality decision.

    VOI(t) = max(0, max_h [E(A(t+h) | D_t) - A(t)] / h)

This uses the **original v2 rule**, not the fixed first-appearance decision and
paired repair-minus-damage diagnostic in the P1 forecasting studies. The
minimum is 2 responses and cap is 16. Stop at the first VOI strictly below
lambda, or at the cap. Confidence controls use the unchanged v2 confidence
family, including the requirement that answer confidence beat OTHER.

The original `configs/v2.json` provides all 19 lambdas, 20 gammas, 48 MC draws,
seed 20261116, horizons 1/2/4/8/16 clipped to include the remaining cap, and all
evaluation settings. P1 heads share the same random streams with each other.
The published M1 traces retain their original random stream. VOI is gain per
additional response; output tokens remain the evaluation cost.

## Computation and checks

P1's count likelihood only needs the correct candidate count, generated count,
valid wrong-answer count and number of wrong groups. The new adapter
precomputes its q/Normal-marginalized evidence on all feasible states through
16 responses. MC continuations then update their posterior by exact lookup,
avoiding repeated quadrature for every synthetic answer sequence.

Before evaluation observations are loaded, each frozen head is checked on the
entire finite state space at successively finer Normal quadrature. The maximum
absolute log-evidence difference must be <=.001. Only numerical integration
resolution can increase; the parameters stay frozen. Native P1 posterior and
lookup results are checked again on the observed prefixes.

Tests cover the native posterior, future labels, reduction to the original M1
MC VOI, label/suffix isolation, chunk invariance, strict threshold semantics,
minimum/cap, provenance and resuming without simulation. The plugin checks
observation order and source fit/bank/baseline/code/trace hashes. The normal
method harness checks stop indices and that returned answers were consumed.

## Evaluation

`pgm v2 method` calls the unchanged `score_families` for every method. It uses:

- the same 3 LLMs x3 benchmarks, dev/test questions and 16-response banks;
- xVerify candidate correctness and cumulative generated + extraction output
  tokens, as in the original study;
- primary 2-fold test cross-fitting (seed 20261117) and separate dev selection;
- matched accuracy at SC16 minus 1 pp, plus 20/30/50% SC16 token budgets;
- a 60-budget, test-oracle frontier summary between SC@2 and SC16;
- question-level paired bootstrap with 10,000 replicates, seed 20261116, and
  exact McNemar against M1.

The selection fold's accuracy constraint is not a guarantee on its held-out
fold. Frontier is an oracle curve summary. Macro scores weight all nine cells
equally and do not supply pooled confidence intervals or multiple-comparison
significance claims. This remains exploratory work on an already inspected
bank. The diagnostic score improvements and actual stopping improvements are
separate empirical questions.

## Commands and outputs

Run from the repository root with its environment active:

```bash
pgm p1-stop trace --workers 4
pgm v2 method --plugin examples/p1_stopping.py
pgm v2 report --methods p1_independent_joint p1_linked_joint p1_independent_mixed p1_linked_mixed
pgm v2 plot --methods M1 asc p1_independent_joint p1_linked_joint p1_independent_mixed p1_linked_mixed --out results/p1_stopping/figures/voi
pgm p1-stop report
```

`p1-stop trace` resumes verified traces. The standard `v2 method` command scores
all eight families and writes `results/v2/methods/p1_*/*.json`. The standard
`v2 report` command above focuses on the four primary P1 families. Without
`--methods`, it includes all eight and splits the static overview into blocks
of four new methods to preserve the existing eight-color palette. The
interactive explorer adds groups for every new method. All eight also remain
in the focused P1 report and can be selected individually in `v2 plot`.

`p1-stop report` writes [results](../results/p1_stopping/README.md), an
[HTML report](../results/p1_stopping/report.html), all operating points,
frontier/macro/paired/numerical CSV tables and three PNG/SVG figures. Its plots
call the original v2 plotting function. The extra figures can also be reproduced:

```bash
pgm v2 plot --methods M1 p1_independent_mixed p1_linked_mixed p1_independent_mixed_v1 p1_linked_mixed_v1 --out results/p1_stopping/figures/weighting
pgm v2 plot --methods M1 p1_independent_mixed p1_independent_mixed_confidence p1_linked_mixed p1_linked_mixed_confidence --out results/p1_stopping/figures/criterion
```

Settings are in `configs/p1_stopping.json`; the complete inherited v2 settings
and implementation hashes are copied to `results/p1_stopping/protocol.json`.
Per-cell locks retain the old fit, bank and baseline hashes. Large trace arrays
live under `artifacts/p1_stopping/<experiment_id>/traces/`.
