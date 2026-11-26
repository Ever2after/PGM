# Evaluation protocol, adding a method, and producing the figures

Every method in this study, whether a PGM arm, a built-in baseline, or a method you add, is scored by one function, `pgm_asc.v2.run.score_families`. Because of that, a new method lands on exactly the same tables and accuracy–token figures as the reported ones.

## 1. Setting

| | |
|---|---|
| Models | Gemma-3-4B-IT, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct (pinned in [configs/models.json](../configs/models.json)) |
| Benchmarks | MATH500, GSM8K, Omni-MATH |
| Cells | 3 models × 3 benchmarks = 9 |
| Splits per benchmark | calibration 128, train 800, dev 200, test (MATH500 500, GSM8K 1,319, Omni-MATH 999) |
| Responses | 16 per question, sampled once with each model's official sampling defaults and stored in a *bank*. Every method replays the same responses in the same order |
| Answers | Parsed by two LLM extraction passes, grouped by exact string (first-observed order); an empty extraction is never an answer |
| Correctness | xVerify-3B-Ia judgement of the returned candidate |
| Cost | **Output tokens** of the consumed responses: generated body + both answer-extraction outputs. Input tokens and feature passes are excluded |

A method may use only the first t responses when it decides to stop at t, and it must return one of their answers. The harness checks both.

## 2. Methods as threshold families

A method is a family indexed by one threshold. For each threshold it outputs, per question:
- a stop index (responses consumed − 1);
- the answer it returns.

Sweeping the threshold traces an accuracy–token curve.

| Method | Threshold | Values swept | Published default | Answer |
|---|---|---|---|---|
| SC@k | k | 1…16 | 16 | majority of the first k |
| ASC (Aggarwal et al. 2023) | β: stop when the Beta top-two posterior P(leader > runner-up) ≥ β | 20 values 0.5–0.999 | 0.95 | majority |
| ESC (Li et al. 2024) | window w: stop at the first window of w identical answers | 2, 3, 4, 5, 6, 8 | 4 | majority |
| ReASC offline (Kim et al. 2026) | β, on votes weighted by max(1, exp(0.7 z)), z = standardized bottom-10% self-certainty; single-response gate from the calibration split | as ASC | 0.95 | weighted majority |
| ReASC online | as offline, gate from a label-free 2-component GMM on the evaluated split's first responses | as ASC | 0.95 | weighted majority |
| PGM v2 arms | λ (VOI arms) or γ (confidence arms) | 19 λ values 0.3–0.0001, or the 20 γ values | none | plurality (MAP for `_map` arms) |

The ASC, ESC and ReASC semantics follow the official ReASC notebooks at `8ee1c76`. Tests check that the v2 implementations (`pgm_asc/v2/baselines.py`) make the same decision as the v1 replication for every question.

## 3. Operating points

A single deployed threshold is chosen **without the evaluated questions' labels**, in two ways:

- **Cross-fitting on test** (primary). The test questions are split randomly in half (seed 20261117). The threshold is chosen on one half and applied to the other, then the halves swap.
- **Dev selection.** The threshold is chosen on the 200 dev questions and applied to all of test. On these benchmarks dev is much easier than test (e.g. Llama MATH500 SC16: dev 86 %, test 59 %), so dev thresholds tend to stop too early on test. That is why cross-fitting is the primary rule.

Selection rules, applied to the method's own curve on the selection questions:

| Rule | Definition |
|---|---|
| `accuracy` (matched accuracy) | Fewest mean tokens with accuracy ≥ SC16 accuracy − 1 pp (on the selection questions); if none qualifies, the most accurate |
| `budget_b`, b ∈ {0.2, 0.3, 0.5} | Highest accuracy with mean tokens ≤ b × SC16 tokens |
| `default` | The published threshold (baselines only) |

The results use the names `cf_accuracy`, `cf_budget_0.3`, … for cross-fitted points, and `accuracy`, `budget_0.3`, … for dev-selected ones.

## 4. Metrics

**Statistical tests.**
- Contrasts are against the primary arm M1, unit = question, on test.
- Δaccuracy and token ratio come with 95 % paired-bootstrap intervals (10,000 replicates, seed 20261116).
- The exact McNemar p-value is computed on the discordant questions.

**Frontier score.** This summarises a whole curve. With B the set of 60 budgets evenly spaced between the test tokens of SC@2 and SC16:

```
Frontier = (1/|B|) Σ_{β∈B} max_{τ : T(τ) ≤ β} Acc(τ)
```

The maximum includes the SC@1 point. The score is larger when the curve sits further up and to the left. It picks the best threshold per budget on test, so it is an oracle summary, but it is applied identically to every method.

## 5. Running the study

```bash
pip install -e '.[test]'
pgm download --only cohorts banks v2   # banks are needed only to rebuild the v2 inputs; `--only v2` is enough to score new methods
pgm v2 prepare --workers 9             # banks -> artifacts/v2/<model>_<dataset>.npz (skipped if present)
pgm v2 run --workers 64                # fit + replay every arm and baseline -> results/v2/cells/*.json, artifacts/v2/traces/
pgm v2 report                          # results/v2/README.md, summary.csv, figures/*.png|svg, report.html
python scripts/report_pdf.py           # optional: results/v2/report.pdf (pip install quickjs weasyprint brotli)
pgm v2 rescore                         # re-run only the evaluation protocol from saved traces (after changing configs/v2.json selection settings)
```

`pgm v2 run` takes about 11 minutes on one 72-core node; most of that is the VOI simulation. On a Slurm cluster, use `sbatch -J pgm-v2 scripts/slurm/v2.sbatch` (add `--models llama8b --datasets math` for a one-cell test).

## 6. Adding a new method

You need `results/v2/cells/*.json` (in the repository) and `artifacts/v2/` (`pgm download --only v2`, 37 MB). The banks are not needed.

### Option A: Python plugin (replays the bank)

```python
# my_method.py
import numpy as np
from pgm_asc.v2.methods import register

@register("my_method", thresholds=[0.5, 0.7, 0.9], default=0.7, label="My method")
def my_method(arrays, threshold, context):
    """arrays: one split, labels removed. Return int arrays [Q]: stop (responses consumed - 1), chosen (answer slot)."""
    ids = arrays["ids"]                    # [Q,16] answer slot per response (first-observed numbering)
    ...
    return stop, chosen
```

**Inputs** (`pgm_asc.v2.methods.SPLIT_INPUTS`):

| Input | Shape | Meaning |
|---|---|---|
| `ids` | [Q,16] | answer slot of each response |
| `valid` | [Q,16] | the response has a non-empty answer |
| `x` | [Q,16,3] | bottom-10 % and lowest windowed self-certainty, log(1 + generated tokens) |
| `tokens` | [Q,16] | output tokens per response |
| `sim_text` | [Q,16,16] | word 3-gram Jaccard similarity of the reasoning texts |
| `sim_hidden` | [Q,16,16] | hidden-state cosine similarity |
| `question_id` | [Q] | question ids |

**`context`**:

| Key | Content |
|---|---|
| `model`, `dataset`, `split` | the cell and split being scored |
| `train`, `calibration` | those splits **with** labels, for fitting or calibrating the method |

A complete example is `examples/plurality_margin.py`, which stops once the leader is k votes ahead. To evaluate it:

```bash
pgm v2 method --plugin examples/plurality_margin.py          # all 9 cells -> results/v2/methods/plurality_margin/*.json
```

### Option B: decisions CSV (method computed elsewhere on the same banks)

```text
model,dataset,split,question_id,threshold,samples,response_index
llama8b,math,test,<question_id>,0.9,3,0
```

| Column | Meaning |
|---|---|
| `samples` | responses consumed (1–16) |
| `response_index` | 0-based index, < `samples`, of a consumed response whose answer is returned |

Provide one row per question of the **dev and test** splits, for every threshold, in every cell you want scored. Question ids are in `artifacts/v2/<model>_<dataset>.npz` (`dev.question_id`, `test.question_id`) and in the cohort files.

```bash
pgm v2 method --csv decisions.csv --name my_method --label "My method"
```

Both options produce, per cell:
- the method's dev and test curves;
- dev-selected and cross-fitted operating points, plus the default if one is given;
- the frontier score;
- paired contrasts against M1.

## 7. Figures and tables

```bash
pgm v2 report                                   # every evaluated method is added after the built-in baselines
pgm v2 report --methods my_method               # only these user methods
pgm v2 plot --methods M1 asc esc my_method --out results/v2/figures/my_method        # one 3×3 figure (PNG + SVG)
pgm v2 plot --methods M1 M0 M1_no_crp --cells llama8b_math qwen7b_omnimath --out fig  # any arms, baselines, user methods, cells
```

**Figure conventions:**
- x = mean output tokens as a share of SC16, on a log scale; y = test accuracy.
- SC@k is drawn in grey, the first method in bold blue, and further methods follow the fixed palette order (up to 8 lines).
- ◆ marks the cross-fitted matched-accuracy point.

`report.html` draws user methods on the 3×3 overview, the curve explorer ("baseline" group), the operating-point scatter and the tables.
With more than four user methods, the static report splits the overview into
additional figures to retain its eight-color palette. The interactive curve
explorer also provides "추가 방법" groups so every user method remains accessible.

**Available method names:**
- `sc`, `asc`, `esc`, `reasc_offline`, `reasc_online`;
- every arm in `configs/v2.json`;
- every evaluated user method.
