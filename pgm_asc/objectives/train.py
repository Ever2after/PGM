"""Same four-parameter M1, different losses; exact compressed count likelihood.

The count likelihood equals the sequential released likelihood, including invalid
responses. Only observed answer hypotheses and OTHER are materialized. This
avoids retaining zero-probability padded hypotheses during gradient computation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import gammaln

from pgm_asc.diagnostics.metrics import targets
from pgm_asc.v2.model import Grid, Spec, init_params, log_prior, log_theta, target_hypothesis, to_json

SPEC = Spec.of("M1")
KEYS = ("log_a", "log_b", "t0", "t1")


def prefix_weights(name):
    if name == "uniform":
        return np.full(16, 1 / 16)
    if name == "v1":
        # Actual v1 code: totals .4/.2/.2/.2 for bands 1-2/3-4/5-8/9-16.
        return np.array([.2, .2] + [.1] * 2 + [.05] * 4 + [.025] * 8)
    raise ValueError(f"Unknown prefix weighting: {name}")


def common_training_subset(arrays):
    keep = targets(arrays)[3]
    if not keep.any():
        raise ValueError("No unique-correct training questions")
    return {k: v[keep] for k, v in arrays.items()}, keep


@dataclass
class Objective:
    kind: str
    weighting: str = "uniform"
    mixing: float | None = None

    @property
    def name(self):
        if self.kind == "joint":
            return "joint"
        if self.kind == "prefix":
            return f"prefix_{self.weighting}"
        if self.kind != "mixed" or self.mixing is None or self.mixing <= 0:
            raise ValueError("Mixed objective needs a positive lambda")
        value = f"{self.mixing:g}".replace(".", "p")
        return f"mixed_{self.weighting}_lambda_{value}"


class Counts:
    def __init__(self, arrays):
        ids, valid = arrays["ids"].astype(int), arrays["valid"]
        self.questions = len(ids)
        if ids.shape[1] != 16:
            raise ValueError("M1 ablation requires 16-response banks")
        count = np.cumsum(np.eye(16)[ids] * valid[..., None], axis=1)
        count = np.concatenate([count, np.zeros((*ids.shape, 1))], axis=2)
        seen = count > 0
        seen[..., 16] = True
        owner, hyp = np.where(seen.reshape(-1, 17))
        self.owner = torch.tensor(owner)
        self.lengths = torch.tensor(seen.sum(2).ravel(), dtype=torch.int64)
        full_target = target_hypothesis(arrays)
        observed_target = np.take_along_axis(count, np.broadcast_to(full_target[:, None, None], (*ids.shape, 1)), 2)[..., 0] > 0
        target = np.where(observed_target, full_target[:, None], 16)
        is_target = hyp == target.ravel()[owner]
        self.target = torch.tensor(np.flatnonzero(is_target))
        assert len(self.target) == self.questions * 16
        self.end_target = self.target.reshape(-1, 16)[:, -1]
        flat = count.reshape(-1, 17)
        nc = flat[owner, hyp]
        total_valid = count.sum(2).ravel()[owner]
        k = (count > 0).sum(2).ravel()[owner] - (nc > 0)
        logfactorial = np.where(count > 0, gammaln(np.maximum(count, 1)), 0.)
        sf = logfactorial.sum(2).ravel()[owner] - logfactorial.reshape(-1, 17)[owner, hyp]
        self.nc = torch.tensor(nc)
        self.wrong_bernoulli = torch.tensor(owner % 16 + 1 - nc)
        self.m = torch.tensor(np.rint(total_valid - nc).astype(int))
        self.k = torch.tensor(k)
        self.sf = torch.tensor(sf)

    def marginal(self, params, grid, indices=None):
        pick = slice(None) if indices is None else indices
        theta_log = log_theta(params, SPEC, grid)
        theta = theta_log.exp()
        rising = torch.lgamma(torch.arange(17)[:, None] + theta[None]) - torch.lgamma(theta)[None]
        ll = (self.nc[pick, None] * grid.lq + self.wrong_bernoulli[pick, None] * grid.l1q
              + self.k[pick, None] * theta_log + self.sf[pick, None] - rising[self.m[pick]])
        return torch.logsumexp(ll + log_prior(params, grid), dim=1)

    def losses(self, params, grid, weighting="uniform", need_prefix=True):
        if not need_prefix:
            return -self.marginal(params, grid, self.end_target).mean() / 16, None
        marginal, nll = self.state_losses(params, grid)
        weights = torch.tensor(prefix_weights(weighting))
        return -marginal[self.end_target].mean() / 16, (nll * weights).sum(1).mean()

    def state_losses(self, params, grid):
        marginal = self.marginal(params, grid)
        maximum = torch.segment_reduce(marginal, "max", lengths=self.lengths)
        sums = torch.segment_reduce((marginal - maximum[self.owner]).exp(), "sum", lengths=self.lengths)
        normalizer = maximum + sums.log()
        nll = (normalizer - marginal[self.target]).reshape(self.questions, 16)
        return marginal, nll


def data_loss(joint, prefix, objective):
    if objective.kind == "joint":
        return joint
    if objective.kind == "prefix":
        return prefix
    return (joint + objective.mixing * prefix) / (1 + objective.mixing)


def fit_one(counts, objective, settings, seed):
    torch.set_num_threads(1)
    grid = Grid()
    params = init_params(SPEC, 0)
    assert tuple(params) == KEYS and sum(p.numel() for p in params.values()) == 4
    rng = np.random.default_rng(seed)
    if seed:
        with torch.no_grad():
            for key in KEYS:
                params[key].copy_(torch.tensor(float(rng.normal(0, settings["initialization_sd"]))))
    initial = to_json(params)
    opt = torch.optim.LBFGS(list(params.values()), lr=1, max_iter=settings["max_iter"],
                          tolerance_grad=settings["tolerance_grad"], tolerance_change=settings["tolerance_change"],
                          history_size=20, line_search_fn="strong_wolfe")
    calls, started = 0, time.perf_counter()

    def components():
        j, c = counts.losses(params, grid, objective.weighting, objective.kind != "joint")
        data = data_loss(j, c, objective)
        ridge = settings["ridge"] * sum(p.square().sum() for p in params.values())
        return j, c, data, data + ridge

    def closure():
        nonlocal calls
        calls += 1
        opt.zero_grad()
        loss = components()[-1]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        return loss

    opt.step(closure)
    opt.zero_grad()
    j, c, data, penalized = components()
    penalized.backward()
    gradient = float(torch.stack([p.grad.abs().max() for p in params.values()]).max())
    state = opt.state[params[KEYS[0]]]
    return {"params": to_json(params), "initial": initial, "restart_seed": seed,
            "joint_nll_per_response": float(j.detach()), "prefix_nll": float(c.detach()) if c is not None else None,
            "data_loss": float(data.detach()), "penalized_loss": float(penalized.detach()),
            "gradient_max_abs": gradient, "iterations": int(state["n_iter"]), "closure_calls": calls,
            "reached_iteration_limit": state["n_iter"] >= settings["max_iter"],
            "seconds": time.perf_counter() - started, "parameters": 4}
