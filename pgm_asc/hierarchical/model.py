"""Exact count likelihood, Gaussian quadrature, and question-shared CRP draws.

The random effect is integrated ONCE for the whole answer sequence. It is never
resampled per response. Observation statistics and inference do not read labels.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import gammaln, logsumexp, roots_hermitenorm

from pgm_asc.diagnostics.predictive import simulate
from pgm_asc.v2.model import Grid, bounded, log_prior, target_hypothesis

KEYS = {
    "M1": ("log_a", "log_b", "t0", "t1"),
    "constant_theta": ("log_a", "log_b", "t0"),
    "P1_independent": ("log_a", "log_b", "t0", "sigma_raw"),
    "P1_linked": ("log_a", "log_b", "t0", "t1", "sigma_raw"),
    "quadratic_theta": ("log_a", "log_b", "t0", "t1", "t2"),
}


def tensors(raw, requires_grad=False):
    return {k: torch.tensor(float(v), requires_grad=requires_grad) for k, v in raw.items()}


def latent_grid(params, variant, nodes=61, sigma_bound=3.):
    """q prior and lognormal theta, with an exactly nested sigma=0 M1."""
    if set(params) != set(KEYS[variant]):
        raise ValueError(f"Wrong parameters for {variant}")
    grid = Grid()
    x = 2 * grid.q - 1
    base = bounded(params["t0"], 6) + (bounded(params["t1"], 6) * x if "t1" in params else torch.zeros_like(x))
    if "t2" in params:
        base = base + bounded(params["t2"], 6) * x.square()
    base = base.clamp(-8, 8)
    if "sigma_raw" in params:
        u, w = roots_hermitenorm(nodes)
        w = w / np.sqrt(2 * np.pi)
        lt = base[:, None] + bounded(params["sigma_raw"], sigma_bound) * torch.tensor(u)[None]
        lp = log_prior(params, grid)[:, None] + torch.tensor(np.log(w))[None]
    else:
        lt, lp = base[:, None], log_prior(params, grid)[:, None]
    # log(theta+r) cumulative sums avoid catastrophic lgamma cancellation.
    logr = torch.tensor(np.r_[-np.inf, np.log(np.arange(1, 16))])
    rising = torch.cat([torch.zeros((1, *lt.shape)),
                        torch.logaddexp(lt[None], logr[:, None, None]).cumsum(0)], 0)
    return grid, lt, lp, rising


class Statistics:
    """All allowed correct-answer hypotheses compressed to sufficient states."""
    def __init__(self, arrays):
        ids, valid = np.asarray(arrays["ids"], int), np.asarray(arrays["valid"], bool)
        if ids.ndim != 2 or ids.shape[1] != 16:
            raise ValueError("Requires a 16-response bank")
        self.questions = len(ids)
        count = np.cumsum(np.eye(16)[ids] * valid[..., None], 1)
        self.count = np.concatenate([count, np.zeros((*ids.shape, 1))], 2)
        self.allowed = self.count > 0
        self.allowed[..., 16] = True
        self.owner, self.hyp = np.where(self.allowed.reshape(-1, 17))
        flat = self.count.reshape(-1, 17)
        nc = flat[self.owner, self.hyp]
        total = self.count.sum(2).ravel()[self.owner]
        k = (self.count > 0).sum(2).ravel()[self.owner] - (nc > 0)
        lf = np.where(self.count > 0, gammaln(np.maximum(self.count, 1)), 0.)
        self.sf = lf.sum(2).ravel()[self.owner] - lf.reshape(-1, 17)[self.owner, self.hyp]
        states = np.array([nc, self.owner % 16 + 1 - nc, total - nc, k], dtype=int).T
        self.states, self.inverse = np.unique(states, axis=0, return_inverse=True)
        self.pairs, inv = np.unique(self.states[:, 2:], axis=0, return_inverse=True)
        self.pair_index = torch.tensor(inv)
        self.state_tensor = torch.tensor(self.states)
        self.map = np.full((self.questions * 16, 17), -1, int)
        self.map[self.owner, self.hyp] = self.inverse
        self.map = self.map.reshape(self.questions, 16, 17)
        self.lengths = torch.tensor(self.allowed.sum(2).ravel())

    def targets(self, arrays):
        h = target_hypothesis(arrays)
        seen = np.take_along_axis(self.count, np.broadcast_to(h[:, None, None], (self.questions, 16, 1)), 2)[..., 0] > 0
        target = np.where(seen, h[:, None], 16).ravel()[self.owner]
        return np.flatnonzero(self.hyp == target).reshape(self.questions, 16)

    def marginal(self, params, variant, nodes=61, sigma_bound=3.):
        grid, lt, lp, rising = latent_grid(params, variant, nodes, sigma_bound)
        # Normal weights are in lp; divide out q prior until the last integral.
        normal_lp = lp - log_prior(params, grid)[:, None]
        pair = torch.logsumexp(torch.tensor(self.pairs[:, 1])[:, None, None] * lt[None]
                               - rising[self.pairs[:, 0]] + normal_lp[None], -1)
        nc, wb = self.state_tensor[:, 0, None], self.state_tensor[:, 1, None]
        state = torch.logsumexp(nc * grid.lq + wb * grid.l1q + pair[self.pair_index]
                                + log_prior(params, grid), -1)
        return state[torch.tensor(self.inverse)] + torch.tensor(self.sf)

    def joint_loss(self, params, variant, arrays, nodes=61, sigma_bound=3.):
        return -self.marginal(params, variant, nodes, sigma_bound)[self.targets(arrays)[:, -1]].mean() / 16


@dataclass
class Inference:
    post: np.ndarray
    statistics: Statistics
    cdf: np.ndarray
    q: np.ndarray
    theta: np.ndarray
    moment_q: np.ndarray
    moment_a: np.ndarray
    moment_b: np.ndarray

    def next_answer(self, i, t):
        count = self.statistics.count[i, t-1, :16]
        p = self.post[i, t-1]
        sid = np.maximum(self.statistics.map[i, t-1], 0)
        a, b, q = self.moment_a[sid], self.moment_b[sid], self.moment_q[sid]
        out = np.zeros(18)
        out[:16] = count * ((p*a).sum() - p[:16]*a[:16]) + p[:16]*q[:16]
        out[16] = (p*b).sum() + p[16]*q[16]
        if not np.isclose(out.sum(), 1, atol=1e-9):
            raise ValueError("Predictive distribution is not normalized")
        return np.maximum(out, 0.)

    def draw(self, i, t, uniforms):
        p = self.post[i, t-1]
        pc = p.cumsum()
        pc[-1] = 1.
        c = np.searchsorted(pc, uniforms, side="right")
        previous = np.r_[0., pc[:-1]][c]
        residual = np.clip((uniforms-previous)/p[c], 0., np.nextafter(1., 0.))
        sid = self.statistics.map[i, t-1, c]
        g = np.empty(len(c), int)
        for s in np.unique(sid):
            mask = sid == s
            g[mask] = np.searchsorted(self.cdf[s], residual[mask], side="right")
        return c, self.q[g], self.theta[g]


@torch.no_grad()
def infer(params, variant, arrays, nodes=61, sigma_bound=3., statistics=None):
    stats = statistics if statistics is not None else Statistics(arrays)
    grid, lt, lp, rising = latent_grid(params, variant, nodes, sigma_bound)
    q = np.broadcast_to(grid.q.numpy()[:, None], lt.shape).ravel().copy()
    theta = lt.exp().numpy().ravel()
    state = stats.states
    latent = (state[:, 0, None, None] * grid.lq.numpy()[None, :, None]
              + state[:, 1, None, None] * grid.l1q.numpy()[None, :, None]
              + state[:, 3, None, None] * lt.numpy()[None]
              - rising.numpy()[state[:, 2]] + lp.numpy()[None]).reshape(len(state), -1)
    evidence = logsumexp(latent, -1)
    prob = np.exp(latent-evidence[:, None])
    moment_q = (prob * q).sum(1)
    denominator = state[:, 2, None] + theta
    moment_a = (prob * ((1-q)/denominator)).sum(1)
    moment_b = (prob * ((1-q)*theta/denominator)).sum(1)
    cdf = prob.cumsum(1)
    cdf[:, -1] = 1.
    value = np.full((stats.questions*16, 17), -np.inf)
    value[stats.owner, stats.hyp] = evidence[stats.inverse] + stats.sf
    post = np.exp(value-logsumexp(value, -1, keepdims=True)).reshape(stats.questions, 16, 17)
    return Inference(post, stats, cdf, q, theta, moment_q, moment_a, moment_b)


def forecast(arrays, inference, prefixes, horizons, simulations=256, seed=20261125, key=""):
    if simulations < 2:
        raise ValueError("At least two simulations required")
    shape = (len(arrays["ids"]), len(prefixes), len(horizons))
    fields = ("future_correct", "leader_change", "future_reference_correct", "gain", "gain_mc_se",
              "repair", "damage", "repair_mc_se", "damage_mc_se", "current_correct_mc")
    out = {k: np.full(shape, np.nan) for k in fields}
    out["next_answer"] = np.full((*shape[:2], 18), np.nan)
    for pi, t in enumerate(prefixes):
        if t == 16:
            continue
        hs = [h for h in horizons if t+h <= 16]
        for i in range(len(arrays["ids"])):
            out["next_answer"][i, pi] = inference.next_answer(i, t)
            if not hs:
                continue
            counts = inference.statistics.count[i, t-1, :16]
            first = np.full(16, 1000)
            for j in range(t):
                if arrays["valid"][i, j]:
                    lab = int(arrays["ids"][i, j])
                    first[lab] = min(first[lab], j-t)
            identity = f"{seed}/{key}/{arrays['question_id'][i]}/{t}".encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity).digest()[:8], "little"))
            latent = inference.draw(i, t, rng.random(simulations))
            samples = simulate(None, counts, first, None, None, hs, rng, simulations,
                               transition_details=True, latent_draws=latent)
            for hi, h in enumerate(horizons):
                if h not in samples:
                    continue
                for name in fields:
                    source = {"gain": "gain_paired", "gain_mc_se": "gain_paired_mc_se"}.get(name, name)
                    out[name][i, pi, hi] = samples[h][source]
    return out


def readable(raw, sigma_bound=3.):
    p = tensors(raw)
    def bound(k, n):
        return float(bounded(p[k], n)) if k in p else 0.
    return {"a": np.exp(bound("log_a", 4)), "b": np.exp(bound("log_b", 4)),
            "theta_intercept": bound("t0", 6), "theta_q_slope": bound("t1", 6),
            "theta_q_quadratic": bound("t2", 6), "sigma_theta": abs(bound("sigma_raw", sigma_bound))}
