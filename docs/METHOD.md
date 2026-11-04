# Method and protocol

An interactive version with diagrams is in [results/report.html](../results/report.html).

## Setting

For a question $q$ the language model is sampled up to 16 times. After each response $y_t$ a policy
either stops and answers, or asks for one more response. All responses come from a fixed bank, so every
method replays the same samples in the same order with the same labels; only the aggregation and the
stopping rule differ. The language models are frozen. The learned part is a head with a few thousand
parameters.

## Inputs per response

| Input | Definition |
|---|---|
| Answer $a_j$ | Final answer from the response or from two short extraction prompts (ReASC procedure). Candidates are exact-string groups in first-observed order. |
| $x_j \in \mathbb{R}^9$ | Teacher-forcing features on the sampled tokens: mean log-prob and entropy of the reasoning span, mean log-prob of the boxed answer, three log lengths, truncation/empty-span flags. |
| $e_j$ | Mean last-layer hidden state over reasoning tokens, L2-normalized. |
| $w_t \in \mathbb{R}^{25}$ | Prefix statistics: 8 vote statistics (counts, leader/runner-up share, margin, answer entropy) and mean/std of four response features over all responses and the leader's responses, plus the truncation rate. |

## The PGM head

**Partitions.** Responses are clustered online by cosine similarity of $e_j$ to running cluster centroids.
Six partitions are kept: $\mathcal P_0$ all-one, $\mathcal P_{1..4}$ with thresholds at the 10/40/70/90 %
quantiles of pairwise train similarities, and $\mathcal P_5$ singletons. $|C_m(j)|$ is the size of the
cluster of response $j$ in partition $m$; duplicate partitions are masked.

**Candidate distribution per partition.**

$$s_j = \mathrm{softplus}(f_\text{sup}(x_j)),\qquad
\ell_m(a) = \sum_{j\le t:\,a_j=a} |C_m(j)|^{\alpha-1} s_j,\qquad
\ell(\texttt{OTHER}) = f_\text{oth}(w_t)$$

$$p_m(a) = \mathrm{softmax}_{a\in\mathcal A_t\cup\{\texttt{OTHER}\}}\,\ell_m(a),\qquad \alpha=\sigma(\beta).$$

$\alpha<1$ discounts responses that fall in large clusters; for singletons or $\alpha=1$ the head is a
confidence-weighted vote. `OTHER` is the probability that the correct answer has not been sampled yet.

**Mixture.** $\pi_m = \mathrm{softmax}_m f_\text{gate}(g_m)$ with 12 structural features $g_m$ per
partition, and $p_t(a)=\sum_m \pi_m p_m(a)$. $f_\text{sup}, f_\text{gate}, f_\text{oth}$ are 2-layer MLPs
(width 32).

**Training.** Negative log-likelihood of the first correct candidate (or `OTHER` if none is correct) at
every prefix $t=1..16$, weighted so that prefixes 1–2, 3–4, 5–8 and 9–16 each carry 0.4. AdamW
(lr 1e-3, weight decay 1e-3), question batches of 32, gradient clipping 1, at most 120 epochs, early
stopping on dev NLL with patience 20. Seeds 42/43/44; seed 42 is primary.

**Calibration and stopping.** A global temperature $T$ is fitted on the calibration cohort. With
$\tilde p_t \propto p_t^{1/T}$ and $\hat a_t=\arg\max_a \tilde p_t(a)$, stop at the first
$t\ge 2$ with $\tilde p_t(\hat a_t)\ge\gamma$ and $\tilde p_t(\hat a_t)>\tilde p_t(\texttt{OTHER})$; otherwise
answer at $t=16$.

**Operating points.** *Selected*: $\gamma\in\{.5,.6,.7,.8,.85,.9,.925,.95,.975,.99,.995,\text{never}\}$
with the fewest dev output tokens among those with dev accuracy $\ge$ dev SC16 − 1 pp. *Fixed*:
$\gamma=0.95$. Test data never selects anything.

## Matched ablations

All arms share inputs, labels, optimizer, budget, calibration and stopping rule.

| Arm | Change |
|---|---|
| `mixture` | Full model |
| `uniform` | Gate fixed to uniform weights |
| `random` | Clusters replaced by random groupings with the same cluster-size multiset at every prefix |
| `alpha1` | $\alpha=1$ (no cluster-size discount); equals `singletons` |
| `single_partition`, `all_one`, `singletons` | One partition only ($\mathcal P_1$, $\mathcal P_0$, $\mathcal P_5$), trained separately |
| `same_head_single` | The trained mixture restricted to $\mathcal P_1$, without retraining |
| `candidate_counts` | MLP on per-candidate vote counts only (no teacher forcing needed) |
| `candidate_confidence` | MLP on per-candidate counts and mean confidence/length |
| `deepsets_confidence` | DeepSets over response features with per-candidate and global sum pooling |
| `deepsets_embedding` | Same, with the reasoning embeddings $e_j$ as additional inputs |

Primary contrasts (per cell, Holm-adjusted over four): mixture vs `alpha1`, `random`,
`candidate_confidence`, `deepsets_embedding`, at the selected operating point, paired by question
(10,000 bootstrap replicates; exact McNemar).

## Baselines

Implemented after the official ReASC notebooks (commit `8ee1c76`).

| Method | Rule |
|---|---|
| `pass1`, `sc16` | First response; majority over 16 |
| `asc` | Beta posterior of the top two vote counts ≥ 0.95 ([Aggarwal et al., 2023](https://aclanthology.org/2023.emnlp-main.761/)) |
| `esc` | Stop at the first window of 4 identical answers ([Li et al., 2024](https://arxiv.org/abs/2401.10480)) |
| `reasc_offline` | Votes weighted by $\max(1, e^{0.7z})$ of bottom-10 % windowed self-certainty; single-response gate; calibrated on the calibration cohort |
| `reasc_online` | Same, calibrated label-free by a two-component GMM on the test questions' first responses |
| `asc_dev_tuned`, `reasc_dev_tuned` | Same algorithms with the threshold chosen on dev by the PGM selection rule |

## Cohorts

| Dataset | Calibration / train / dev | Test |
|---|---|---|
| MATH | 128 (79 original-train, 49 original-test; disjoint from test) / 800 / 200 original-train questions | MATH500 |
| GSM8K | 128 / 800 / 200 from the official train split (seed 20261103; test duplicates removed) | All 1,319 official test questions |
| Omni-MATH | 128 / 800 / 200 from the full release (`KbsdJames/Omni-MATH` @ `40ba231`) after removing every test question by normalized text (seed 20261103) | 999 distinct questions of the ReASC Omni archive |

Responses: at most 1,024 new tokens (2,048 for Omni-MATH), each model's official sampling defaults
(`configs/models.json`). Correctness: pinned xVerify-3B-Ia, greedy. Cost: generated tokens plus both
extraction outputs; input tokens and the teacher-forcing pass are reported separately in the
evaluation files and excluded from the main cost.

## Cells

16 cells = 4 models × {`math`, `gsm8k`, `omnimath`, `math_to_omnimath`}. The first three are fitted
and evaluated on the same dataset. `math_to_omnimath` replays the model's MATH policy (heads,
calibrators, thresholds and ReASC offline calibration) on the Omni-MATH test bank without any Omni
labels; it shares the test bank with the in-domain `omnimath` cell.
