# PGM v2: a small generative model of sampled answers

This document describes the v2 model, its variants (arms), and why each part is there. The evaluation protocol shared by all methods is in [PROTOCOL.md](PROTOCOL.md), and the results are in [results/v2/README.md](../results/v2/README.md). An interactive version with equations and charts is [results/v2/report.html](../results/v2/report.html); a PDF version is [results/v2/report.pdf](../results/v2/report.pdf).

v2 restarts from the simplest model that is still a probabilistic graphical model. It keeps from v1 only what v1 validated:
- the shared response banks: 16 samples per question with xVerify labels;
- the baselines and the cost definition;
- the "OTHER" hypothesis (the correct answer has not been sampled yet);
- threshold selection without test labels.

It drops the learned neural heads and the semantic partitions, because v1 found no gain from them ([v1 summary](v1/README.md)).

## Lessons carried over from v1

| v1 finding | v2 consequence |
|---|---|
| Semantic partitions do not beat size-matched random grouping (1/48 Holm-significant contrasts) | No partition mixture. Redundancy between reasoning paths is one explicit node, tested against a permuted control |
| Response confidence mostly tracks question difficulty (within-question AUC 0.53–0.69) | Confidence is a child of the latent difficulty q, with an optional per-response correctness effect. Both are ablated |
| How wrong answers concentrate depends on difficulty | Wrong answers follow a Chinese restaurant process (CRP) whose concentration θ(q) depends on q |
| Hard questions spend most tokens at the 16-sample cap, often with no correct answer (Omni-MATH: 70–83% of questions hit the cap) | Futility stopping from the model's own predictive distribution (VOI) |
| DeepSets on embeddings gave no gain | Hidden states enter only as an alternative similarity for the redundancy node (ablation) |

## Model

For one question with responses j = 1..16 in sampling order:

```
q    ~ Beta(a, b)                              latent accuracy of the model on this question (difficulty)
c    ~ H                                       the correct answer (continuous base measure)
z_j  ~ Bernoulli(q)                            response j is correct
a_j  = c                          if z_j = 1
a_j  ~ CRP(θ(q)) over wrong answers  if z_j = 0:  an earlier wrong answer w with prob n_w / (m_j + θ),
                                                  a new wrong answer with prob θ / (m_j + θ)
log θ(q) = t0 + t1 (2q − 1)
```

Here n_w is the number of earlier responses with wrong answer w, and m_j is the number of earlier wrong responses. A small θ means wrong answers pile up on one or two values. Empty extractions count as wrong responses (z_j = 0) that join no cluster. The final model **M1** has four parameters: a, b, t0, t1.

**Likelihood of a hypothesis.** Let D_t = a_1..t, n_a the votes for candidate a, and n_k the other votes. Then

```
p(D_t | c = a, q) ∝ q^{n_a} (1 − q)^{t − n_a} · θ(q)^{K_{−a}} ∏_{k≠a} Γ(n_k) · Γ(θ) / Γ(θ + t − n_a)
```

The second factor is the Ewens probability of the wrong-answer partition, and K_{−a} is the number of distinct wrong answers. The base measure H contributes the same factor to every hypothesis and cancels.

**Posterior.** P(c = a | D_t) is exact:
- q is integrated on an 80-node logit grid, q = σ(z), z ∈ [−7, 7];
- c is enumerated over the observed candidates plus OTHER, the hypothesis that the correct answer has not been sampled yet.

**Fitting.** The parameters maximise the likelihood of the training banks including the labelled correct answer, Σ_i log ∫ p(q) p(a_{i,1:16}, c*_i | q) dq, where c* is the correct candidate (or OTHER if no response is correct). Training uses 800 questions and L-BFGS, and takes a few seconds per cell. a, b and θ are soft-bounded so that near-degenerate cells stay finite.

## Answer and stopping

The model decides **when to stop**. The **answer** is the plurality answer, with ties broken by the posterior:

```
â_t = argmax_{a : n_a = max_b n_b} P(c = a | D_t),        A_t = P(c = â_t | D_t)
VOI_t = max_{h ∈ {1, 2, 4, 8, 16 − t}} ( E[A_{t+h} | D_t] − A_t ) / h
stop at t  ⇔  t ≥ 2  and  VOI_t < λ          (else continue; stop at 16)
```

E[A_{t+h} | D_t] is estimated with 48 Monte Carlo continuations:
1. Draw (c, q) from the joint posterior.
2. Generate h further answers from the model.
3. Recompute the posterior exactly.

VOI is the best expected gain in accuracy per extra sample. Both easy questions (already certain) and hopeless questions (more samples will not help) stop early. Count-based rules (ASC, ESC, ReASC) have no predictive distribution, so they cannot compute it. λ is the single knob, swept to trace the accuracy–token curve.

The posterior argmax answer ("MAP answer") is kept as an ablation. On hard sets the fitted prior puts most of its mass near q ≈ 0. There the CRP over-predicts how strongly wrong answers cluster, so the MAP answer prefers singletons over a 3-vote answer. Before the plurality answer was introduced, this dropped the 16-sample accuracy on Omni-MATH below SC16 (Gemma 28.2 vs 31.6, Llama 13.2 vs 17.2).

## Variants (arms)

| Arm | Change from M1 | Question it answers |
|---|---|---|
| **M1** (primary) | q, θ(q), CRP, OTHER; VOI stop; plurality answer | |
| M0 | Confidence stop instead of VOI: stop when P(c = â \| D) ≥ γ and it beats OTHER | Does the predictive distribution help? |
| M1_map | Posterior argmax answer | Is the posterior a better answer selector than plurality? |
| M1_no_other | No OTHER hypothesis | Is the unseen-answer mass needed? |
| M1_theta_const | t1 = 0 | Does wrong-answer concentration need to depend on difficulty? |
| M1_no_crp | Each wrong answer has a fixed probability ε (no clustering) | Is wrong-answer clustering needed? |
| M2, M2_map | + confidence node: x_j ~ N(b0 + b1·logit(q)/7 + b2·z_j, s²) on the bottom-10% self-certainty | Does self-certainty help, as difficulty or as evidence? |
| M1_copy | + copy node: a_j repeats the answer of its most similar earlier response k* with prob ρ·u_j^κ, u_j = train-pair CDF of sim(j, k*) | Does discounting redundant reasoning paths help? |
| M1_copy_hidden | Copy node on hidden-state cosine instead of 3-gram Jaccard | Are embeddings needed? |
| M1_copy_perm | Copy node on similarities permuted within each question | Control for M1_copy |
| M3 | + confidence + copy | Full model |

The arms are defined in [configs/v2.json](../configs/v2.json) and implemented in `pgm_asc/v2/model.py` (likelihood, posterior, fit) and `pgm_asc/v2/voi.py` (VOI).

## What the ablations show

Mean frontier score over the 9 cells (see [PROTOCOL.md](PROTOCOL.md#metrics)):

| Change | Frontier change | Reading |
|---|---:|---|
| Remove CRP (independent wrong answers) | **−3.9 pp** | Needed: modelling how wrong answers cluster is the core of the graph |
| VOI → confidence (γ) stop | −1.1 pp | Needed: the predictive distribution gives a better stopping rule |
| Remove OTHER | −0.4 pp | Small; matters on hard sets (Llama Omni-MATH +1.8 pp for M1, p = 0.01) |
| + confidence node | −0.7 pp | Not needed; the node is learned as a difficulty signal and makes the posterior over-confident |
| + copy node (text, hidden or permuted) | ±0.0 pp | Not needed; real and permuted similarities give the same result, so reasoning-path redundancy carries no signal here, and neither do embeddings |
| θ constant, MAP answer | ≤ 0.1 pp | Optional |
