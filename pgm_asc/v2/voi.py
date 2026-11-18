"""Value of sampling more, from the model's own posterior predictive (Monte Carlo).

At prefix t the expected accuracy of answering now is A_t = P(c = answer_t | D_t), where answer_t is the plurality
answer (ties broken by the posterior) or, for decision = "map", the posterior argmax. Draw (c, q) from the joint
posterior, simulate h further answers from the generative model (no copies, confidences unobserved), update the
posterior exactly with the count likelihood, and average the new maximum: E[A_{t+h} | D_t]. The stopping statistic is
the best expected accuracy gain per extra sample, VOI_t = max_h (E[A_{t+h}] - A_t) / h over a few horizons h.
A generative model is what makes this computable: count-based rules (ASC, ESC, ReASC) have no predictive distribution.

Future label layout per state: slots 0..15 (observed candidates), 16 = the correct answer if it is still unseen,
17..31 = new wrong answers, 32 = OTHER (correct answer still unseen at t + h).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import gammaln, logsumexp

from pgm_asc.v2 import CONFIG, OTHER, SLOTS
from pgm_asc.v2.model import Grid, Spec, log_theta

N_LABELS = 33
NEWC, OTHER_F = 16, 32


def _count_ll(n, total, lq, l1q, lt, lgt, log_eps, crp):
    """log P(counts | c = label, q) for every label (count model), shape [..., labels, G]; ``n`` [..., labels]."""
    nc = n[..., None]
    rest = total[..., None, None] - nc
    base = nc * lq + rest * l1q
    if not crp:
        return base + rest * log_eps
    k = (n > 0).sum(-1)[..., None, None] - (nc > 0)
    sl = np.where(n > 0, gammaln(np.maximum(n, 1)), 0.).sum(-1)[..., None, None] - np.where(nc > 0, gammaln(np.maximum(nc, 1)), 0.)
    # log Gamma(theta + m) - log Gamma(theta) from a table over m = 0..32
    rising = np.take(lgt, np.rint(rest[..., 0]).astype(int), axis=0)          # [..., labels, G]
    return base + k * lt + sl - rising


def state_values(post, joint, allowed, chosen, counts0, total0, t, cap, lq, l1q, q, lt, theta, lgt, log_eps, spec, rng, sims, horizons):
    """VOI at one prefix: max_h (E[A_{t+h}] - A_t) / h, A = posterior probability that the answer we would give is correct."""
    crp = spec.wrong == "crp"
    a_t = post[chosen] if chosen >= 0 else 0.
    hyp_ok = np.flatnonzero(allowed)
    if not len(hyp_ok):
        return 1.
    flat = (joint[hyp_ok] - joint[hyp_ok].max()).ravel()
    p = np.exp(flat)
    p /= p.sum()
    draw = rng.choice(len(p), size=sims, p=p)
    g_s = draw % joint.shape[1]
    h_s = hyp_ok[draw // joint.shape[1]]
    c_s = np.where(h_s == OTHER, NEWC, h_s)
    q_s, th_s = q[g_s], theta[g_s]
    counts = np.zeros((sims, N_LABELS))
    counts[:, :SLOTS] = counts0
    nxt = np.full(sims, 17)
    rows = np.arange(sims)
    base = np.empty((N_LABELS, joint.shape[1]))
    base[:SLOTS] = joint[:SLOTS]
    base[SLOTS:] = joint[OTHER]
    ll0 = _count_ll(counts[:1], np.array([total0]), lq, l1q, lt, lgt, log_eps, crp)[0]  # [labels, G]
    hmax = cap - t
    marks = sorted({h for h in horizons if h < hmax} | {hmax})
    best = 0.
    for step in range(1, hmax + 1):
        correct = rng.random(sims) < q_s
        wrong = counts.copy()
        wrong[rows, c_s] = 0
        wrong[:, OTHER_F] = 0
        m = wrong.sum(1)
        if crp:
            p_new = th_s / (m + th_s)
        else:
            p_new = np.clip(1 - (wrong > 0).sum(1) * np.exp(log_eps), 0, 1)
        new = rng.random(sims) < p_new
        cum = np.cumsum(wrong, 1) / np.maximum(m, 1)[:, None]
        old = (rng.random(sims)[:, None] < cum).argmax(1)
        label = np.where(correct, c_s, np.where(new | (m == 0), nxt, old))
        nxt = nxt + (~correct & (new | (m == 0)))
        counts[rows, np.minimum(label, OTHER_F - 1)] += 1
        if step in marks:
            ll = _count_ll(counts, np.full(sims, total0 + step), lq, l1q, lt, lgt, log_eps, crp)   # [S, labels, G]
            logp = logsumexp(base + ll - ll0, axis=-1)                                           # [S, labels]
            ok = counts > 0                      # observed candidates (all valid: counts only count valid answers)
            ok[:, OTHER_F] = spec.other
            logp = np.where(ok, logp, -np.inf)
            logp -= logsumexp(logp, axis=1, keepdims=True)
            pick = ok[:, :OTHER_F]
            if spec.decision == "plurality":
                c = np.where(pick, counts[:, :OTHER_F], -1)
                pick = pick & (c == c.max(1, keepdims=True))
            cand = np.where(pick, logp[:, :OTHER_F], -np.inf)
            a_h = np.exp(cand.max(1)).mean()
            best = max(best, (a_h - a_t) / step)
    return best


def voi(params: dict, spec: Spec, arrays: dict, posterior, key: str = "") -> np.ndarray:
    """VOI for every question and prefix t = 1..15: [Q,15] (prefix 16 is the cap)."""
    cfg, cap = CONFIG["voi"], CONFIG["stopping"]["cap"]
    grid = Grid()
    lq, l1q, q = grid.lq.numpy(), grid.l1q.numpy(), grid.q.numpy()
    if spec.wrong == "crp":
        lt = log_theta(params, spec, grid).detach().numpy()
        theta = np.exp(lt)
        lgt = gammaln(theta[None, :] + np.arange(N_LABELS)[:, None]) - gammaln(theta)[None, :]
        log_eps = 0.
    else:
        lt = theta = np.zeros_like(q)
        lgt = None
        log_eps = float(torch.nn.functional.logsigmoid(params["eps_logit"]))
    ids, valid = arrays["ids"].astype(int), arrays["valid"]
    out = np.zeros((len(ids), cap - 1))
    for i in range(len(ids)):
        rng = np.random.default_rng([cfg["seed"], i, sum(map(ord, key))])
        counts = np.zeros(SLOTS)
        for t in range(1, cap):
            if valid[i, t - 1]:
                counts[ids[i, t - 1]] += 1
            out[i, t - 1] = state_values(posterior.post[i, t - 1], posterior.joint[i, t - 1], posterior.allowed[i, t - 1],
                                         int(posterior.chosen[i, t - 1]),
                                         counts, counts.sum(), t, cap, lq, l1q, q, lt, theta, lgt, log_eps, spec, rng,
                                         cfg["simulations"], cfg["horizons"])
    return out
