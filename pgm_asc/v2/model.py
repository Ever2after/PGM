"""The v2 generative model of one question's sampled answers.

Graph (per question, responses j = 1..16 in sampling order)::

    q ~ Beta(a, b)                                   latent accuracy of the model on this question (difficulty)
    c ~ H (continuous base measure)                  the correct answer
    z_j | q ~ Bernoulli(q)                           response j is correct
    copy_j | sim ~ Bernoulli(r_j)                    response j re-traces an earlier reasoning path k*_j
    a_j = a_{k*_j}                    if copy_j      (k*_j = most similar earlier response)
    a_j = c                           elif z_j
    a_j ~ CRP(theta(q)) over wrong answers otherwise  (wrong answers cluster; theta depends on difficulty)
    x_j | q, z_j ~ N(b0 + b1 d(q) + b2 z_j, s^2)     per-response confidence (difficulty and/or evidence),
                                                     d(q) = logit(q) / 7 in [-1, 1] on the grid

with r_j = rho * u_j^kappa, u_j = train-pair CDF of max_k<j sim(j, k). Integrating q on a fixed grid and enumerating
c over the observed candidates plus OTHER (not yet observed) gives the exact posterior P(c | a_1..t, x_1..t).
The base measure H contributes the same factor to every hypothesis and cancels. All parameters (<= 10) are fitted
by maximum likelihood of the training banks including the labelled correct answer (P(a, x, c*)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from pgm_asc.v2 import CONFIG, OTHER, SLOTS

torch.set_default_dtype(torch.float64)
H_ALL = SLOTS + 1
LOG_2PI = math.log(2 * math.pi)


@dataclass(frozen=True)
class Spec:
    theta: str = "q"        # "q": log theta = t0 + t1 (2q - 1);  "const": t1 = 0  (t0, t1 soft-bounded to +-6)
    wrong: str = "crp"      # "crp": wrong answers from a Chinese restaurant process;  "flat": each wrong answer has prob eps
    x: str = "both"         # "none" | "both" | "difficulty" (b2 = 0) | "evidence" (b1 = 0)
    copy: str = "text"      # "none" | "text" | "hidden" | "perm" (text similarity permuted within question)
    other: bool = True      # keep the OTHER hypothesis in the posterior and the stopping rule
    stop: str = "voi"       # "confidence" (max posterior >= gamma) | "voi" (value of one more sample < lambda)
    decision: str = "plurality"  # answer = most frequent observed answer, ties broken by the posterior | "map" (posterior argmax)

    @classmethod
    def of(cls, arm: str) -> "Spec":
        a = CONFIG["arms"][arm]
        return cls(a["theta"], a["wrong"], a["x"], a["copy"], a["other"], a["stop"], a.get("decision", "plurality"))


class Grid:
    def __init__(self, size=None, lo=None, hi=None):
        size = size or CONFIG["grid_size"]
        lo, hi = CONFIG["grid_logit_range"] if lo is None else (lo, hi)
        z = torch.linspace(lo, hi, size)
        self.q = torch.sigmoid(z)
        self.lq, self.l1q = torch.nn.functional.logsigmoid(z), torch.nn.functional.logsigmoid(-z)
        self.log_dq = self.lq + self.l1q + math.log(float(z[1] - z[0]))
        self.d = z / max(abs(lo), abs(hi))          # difficulty scale for the confidence mean, in [-1, 1]


def bounded(v, limit):
    """Smooth clamp to (-limit, limit): keeps every fitted parameter finite on near-degenerate cells."""
    return limit * torch.tanh(v / limit)


S_FLOOR = .05   # floor on the confidence noise (standardized units), so the Gaussian cannot collapse


# ------------------------------------------------------------------ parameters
def init_params(spec: Spec, d: int) -> dict:
    p = {"log_a": 0., "log_b": 0.}
    if spec.wrong == "crp":
        p["t0"] = 0.
        if spec.theta == "q":
            p["t1"] = 0.
    else:
        p["eps_logit"] = -2.          # each wrong answer value has probability eps = sigmoid(eps_logit) < 1
    if spec.x != "none":
        p["b0"], p["log_s"] = [0.] * d, [0.] * d
        if spec.x in ("both", "difficulty"):
            p["b1"] = [0.] * d
        if spec.x in ("both", "evidence"):
            p["b2"] = [0.] * d
    if spec.copy != "none":
        p["rho"], p["log_kappa"] = -2., math.log(2.)
    return {k: torch.tensor(v, requires_grad=True) for k, v in p.items()}


def to_json(params: dict) -> dict:
    return {k: (v.detach().tolist()) for k, v in params.items()}


def from_json(values: dict) -> dict:
    return {k: torch.tensor(v) for k, v in values.items()}


def readable(params: dict) -> dict:
    """Fitted values on their natural scale (for reports)."""
    p = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in params.items()}
    out = {"a": float(bounded(p["log_a"], 4.).exp()), "b": float(bounded(p["log_b"], 4.).exp())}
    if "t0" in p:
        t1 = bounded(p["t1"], 6.) if "t1" in p else 0.
        for q in (.1, .5, .9):
            out[f"theta(q={q})"] = float((bounded(p["t0"], 6.) + t1 * (2 * q - 1)).clamp(-8, 8).exp())
    if "eps_logit" in p:
        out["eps"] = float(torch.sigmoid(p["eps_logit"]))
    for k in ("b0", "b1", "b2"):
        if k in p:
            out[k] = p[k].tolist()
    if "log_s" in p:
        out["s"] = (S_FLOOR + p["log_s"].exp()).tolist()
    if "rho" in p:
        out["rho"], out["kappa"] = float(torch.sigmoid(p["rho"])), float(p["log_kappa"].exp())
    return out


def n_parameters(params: dict) -> int:
    return int(sum(v.numel() for v in params.values()))


# ------------------------------------------------------------------ inputs
def transforms(train: dict, spec: Spec, x_index) -> dict:
    """Train-split statistics: feature standardization and the pairwise-similarity CDF."""
    x = train["x"][..., x_index].reshape(-1, len(x_index)).astype(np.float64)
    out = {"x_mean": x.mean(0).tolist(), "x_std": np.maximum(x.std(0), 1e-6).tolist(), "x_index": list(x_index)}
    if spec.copy != "none":
        sim = similarity(train, spec)
        valid = train["valid"]
        iu = np.triu_indices(sim.shape[1], 1)
        pair = sim[:, iu[0], iu[1]][valid[:, iu[0]] & valid[:, iu[1]]]
        pair = np.sort(pair[np.isfinite(pair)])
        out["sim_reference"] = pair[np.linspace(0, len(pair) - 1, min(len(pair), 4001)).astype(int)].tolist()
    return out


def similarity(arrays: dict, spec: Spec) -> np.ndarray:
    if spec.copy == "hidden":
        return arrays["sim_hidden"].astype(np.float64)
    sim = arrays["sim_text"].astype(np.float64)
    if spec.copy == "perm":
        # Same similarity values, randomly re-assigned to responses within each question: keeps the distribution,
        # breaks the link between reasoning similarity and the answers.
        rng = np.random.default_rng(CONFIG["voi"]["seed"])
        perm = np.argsort(rng.random(sim.shape[:2]), axis=1)
        sim = np.take_along_axis(np.take_along_axis(sim, perm[:, :, None], 1), perm[:, None, :], 2)
    return sim


@dataclass
class Batch:
    ids: torch.Tensor          # [Q,16] long
    valid: torch.Tensor        # [Q,16] bool
    prev: torch.Tensor         # [Q,16,17] counts of slot k among earlier valid responses (column OTHER = 0)
    prev_valid: torch.Tensor   # [Q,16]
    n_same: torch.Tensor       # [Q,16] earlier valid responses with the same answer
    seen: torch.Tensor         # [Q,16,16] slot k is a valid candidate observed by prefix t
    x: torch.Tensor | None     # [Q,16,D]
    u: torch.Tensor | None     # [Q,16] similarity CDF of the most similar earlier valid response
    match: torch.Tensor | None # [Q,16] response j repeats the answer of that response


def make_batch(arrays: dict, spec: Spec, tf: dict, rows=None) -> Batch:
    rows = slice(None) if rows is None else rows
    ids_np = arrays["ids"][rows].astype(np.int64)
    valid_np = arrays["valid"][rows]
    q, n = ids_np.shape
    onehot = np.zeros((q, n, H_ALL))
    onehot[np.arange(q)[:, None], np.arange(n)[None, :], ids_np] = valid_np
    cum = np.cumsum(onehot, 1)
    prev = cum - onehot
    seen = cum[:, :, :SLOTS] > 0
    x = u = match = None
    if spec.x != "none":
        xi = tf["x_index"]
        x = torch.tensor((arrays["x"][rows][..., xi] - np.asarray(tf["x_mean"])) / np.asarray(tf["x_std"]))
    if spec.copy != "none":
        sim = similarity(arrays, spec)[rows]
        earlier = np.tril(np.ones((n, n), bool), -1)[None] & valid_np[:, None, :]
        masked = np.where(earlier & np.isfinite(sim), sim, -np.inf)
        kstar = masked.argmax(2)
        smax = np.take_along_axis(masked, kstar[:, :, None], 2)[..., 0]
        ref = np.asarray(tf["sim_reference"])
        u_np = np.where(np.isfinite(smax), np.searchsorted(ref, smax, side="right") / len(ref), 0.)
        has = np.isfinite(smax)
        match_np = has & valid_np & (np.take_along_axis(ids_np, kstar, 1) == ids_np)
        u, match = torch.tensor(np.clip(u_np, 1e-6, 1.)), torch.tensor(match_np)
    return Batch(torch.tensor(ids_np), torch.tensor(valid_np), torch.tensor(prev), torch.tensor(np.cumsum(valid_np, 1) - valid_np, dtype=torch.float64),
                 torch.tensor(np.take_along_axis(prev, ids_np[:, :, None], 2)[..., 0]), torch.tensor(seen), x, u, match)


# ------------------------------------------------------------------ likelihood
def log_theta(params: dict, spec: Spec, grid: Grid) -> torch.Tensor:
    t = bounded(params["t0"], 6.) + (bounded(params["t1"], 6.) * (2 * grid.q - 1) if spec.theta == "q" else 0. * grid.q)
    return t.clamp(-8, 8)


def log_prior(params: dict, grid: Grid) -> torch.Tensor:
    a, b = bounded(params["log_a"], 4.).exp(), bounded(params["log_b"], 4.).exp()
    w = (a - 1) * grid.lq + (b - 1) * grid.l1q + grid.log_dq
    return w - torch.logsumexp(w, 0)


def factors(params: dict, spec: Spec, batch: Batch, hyp: torch.Tensor, grid: Grid) -> torch.Tensor:
    """log p(a_j, x_j | a_<j, c = hyp, q = grid) for every response: [Q,16,H,G]."""
    qn, n = batch.ids.shape
    h = hyp.shape[1]
    valid = batch.valid[:, :, None, None]
    is_c = ((batch.ids[:, :, None] == hyp[:, None, :]) & batch.valid[:, :, None])[..., None]   # [Q,16,H,1]
    lq, l1q = grid.lq, grid.l1q
    if spec.wrong == "crp":
        lt = log_theta(params, spec, grid)                                                    # [G]
        prev_h = batch.prev.gather(2, hyp[:, None, :].expand(qn, n, h))                       # [Q,16,H]
        m_prev = (batch.prev_valid[:, :, None] - prev_h)[..., None]                           # [Q,16,H,1]
        same = batch.n_same[:, :, None, None]
        num = torch.where(same > 0, same.clamp(min=1).log(), lt)                             # [Q,16,1,G]
        wrong = l1q + num - torch.log(m_prev + lt.exp())
    else:
        wrong = (l1q + torch.nn.functional.logsigmoid(params["eps_logit"])).expand(qn, n, h, -1)
    f = torch.where(is_c, lq, torch.where(valid, wrong, l1q))
    if spec.copy != "none":
        r = (torch.sigmoid(params["rho"]) * batch.u ** params["log_kappa"].exp()).clamp(1e-10, 1 - 1e-10)[:, :, None, None]
        stay = torch.log1p(-r) + f
        f = torch.where(batch.match[:, :, None, None], torch.logaddexp(r.log().expand_as(stay), stay), stay)
    if spec.x != "none":
        s = S_FLOOR + params["log_s"].exp()                                                   # [D]
        mean = params["b0"][:, None] + (params["b1"][:, None] * grid.d if "b1" in params else 0.)   # [D,G]
        diff = batch.x[..., None] - mean                                                      # [Q,16,D,G]
        norm = -(s.log().sum() + .5 * LOG_2PI * len(s))
        lp0 = -.5 * ((diff / s[:, None]) ** 2).sum(2) + norm                                  # [Q,16,G]
        if "b2" in params:
            lp1 = -.5 * (((diff - params["b2"][:, None]) / s[:, None]) ** 2).sum(2) + norm
        else:
            lp1 = lp0
        f = f + torch.where(is_c, lp1[:, :, None, :], lp0[:, :, None, :])
    return f


def target_hypothesis(arrays: dict) -> np.ndarray:
    """The labelled correct answer: the most frequent correct candidate (first observed on ties), else OTHER."""
    ids, valid, correct = arrays["ids"], arrays["valid"], arrays["correct"]
    counts = np.zeros((len(ids), SLOTS))
    np.add.at(counts, (np.arange(len(ids))[:, None].repeat(ids.shape[1], 1), ids), valid)
    score = np.where(correct, counts, -1)
    best = score.argmax(1)
    return np.where(score.max(1) > 0, best, OTHER)


def neg_log_likelihood(params, spec, batch, target, grid):
    f = factors(params, spec, batch, target[:, None], grid)[:, :, 0, :]                     # [Q,16,G]
    return -torch.logsumexp(log_prior(params, grid) + f.sum(1), -1).mean()


def fit(arrays: dict, spec: Spec, tf: dict, max_iter: int = 300) -> tuple[dict, dict]:
    """Maximum likelihood of P(a_1..16, x_1..16, c*) on the training split (L-BFGS, a 1e-4 ridge for stability)."""
    torch.set_num_threads(1)
    grid = Grid()
    batch = make_batch(arrays, spec, tf)
    target = torch.tensor(target_hypothesis(arrays))
    params = init_params(spec, len(tf["x_index"]))
    opt = torch.optim.LBFGS(list(params.values()), lr=1, max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")
    with torch.no_grad():
        start = float(neg_log_likelihood(params, spec, batch, target, grid))

    def closure():
        opt.zero_grad()
        loss = neg_log_likelihood(params, spec, batch, target, grid) + 1e-4 * sum((v ** 2).sum() for v in params.values())
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        final = float(neg_log_likelihood(params, spec, batch, target, grid))
    if not math.isfinite(final):
        raise FloatingPointError("Non-finite likelihood after fitting")
    summary = {"nll_per_question_start": start, "nll_per_question": final, "parameters": n_parameters(params),
               "questions": int(len(target)), "iterations": int(opt.state[opt._params[0]]["n_iter"])}
    return {k: v.detach() for k, v in params.items()}, summary


# ------------------------------------------------------------------ posterior
def prefix_counts(arrays: dict) -> np.ndarray:
    """[Q,16,16] number of valid responses with answer slot k among the first t."""
    ids, valid = arrays["ids"].astype(int), arrays["valid"]
    q, n = ids.shape
    onehot = np.zeros((q, n, SLOTS))
    onehot[np.arange(q)[:, None], np.arange(n)[None, :], ids] = valid
    return np.cumsum(onehot, 1)



@dataclass
class Posterior:
    post: np.ndarray       # [Q,16,17] P(c = slot | prefix t), column OTHER
    joint: np.ndarray      # [Q,16,17,G] log p(prefix t, c = slot, q = grid node), unnormalized
    allowed: np.ndarray    # [Q,16,17] hypotheses in the posterior support at prefix t
    chosen: np.ndarray     # [Q,16] answered candidate (plurality or posterior argmax; -1 if none yet)
    confidence: np.ndarray # [Q,16] its posterior probability


def posterior(params: dict, spec: Spec, arrays: dict, tf: dict, chunk: int = 128, keep_joint=True) -> Posterior:
    torch.set_num_threads(1)
    grid = Grid()
    q = len(arrays["ids"])
    outs = []
    with torch.no_grad():
        lp = log_prior(params, grid)
        for start in range(0, q, chunk):
            rows = slice(start, min(q, start + chunk))
            batch = make_batch(arrays, spec, tf, rows)
            hyp = torch.arange(H_ALL).expand(len(batch.ids), -1)
            joint = torch.cumsum(factors(params, spec, batch, hyp, grid), 1) + lp
            allowed = torch.cat([batch.seen, torch.full((*batch.seen.shape[:2], 1), spec.other)], 2)
            outs.append((joint.numpy() if keep_joint else None, allowed.numpy(), torch.logsumexp(joint, -1).numpy()))
    joint = np.concatenate([o[0] for o in outs]) if keep_joint else None
    allowed = np.concatenate([o[1] for o in outs])
    marginal = np.concatenate([o[2] for o in outs])
    logits = np.where(allowed, marginal, -np.inf)
    mx = logits.max(2, keepdims=True)
    post = np.where(allowed, np.exp(logits - np.where(np.isfinite(mx), mx, 0)), 0.)
    post /= np.maximum(post.sum(2, keepdims=True), 1e-300)
    ok = allowed[:, :, :SLOTS]
    if spec.decision == "plurality":
        counts = prefix_counts(arrays)
        counts = np.where(ok, counts, -1)
        ok = ok & (counts == counts.max(2, keepdims=True))
    cand = np.where(ok, post[:, :, :SLOTS], -1.)
    chosen = np.where(allowed[:, :, :SLOTS].any(2), cand.argmax(2), -1)
    confidence = np.where(chosen >= 0, np.take_along_axis(post, np.maximum(chosen, 0)[..., None], 2)[..., 0], 0.)
    return Posterior(post, joint, allowed, chosen, confidence)
