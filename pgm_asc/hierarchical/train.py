"""Controlled joint-likelihood fitting with common starts and fixed cohorts."""
from __future__ import annotations

import time
import numpy as np
import torch

from pgm_asc.hierarchical.model import KEYS, Statistics, infer, tensors
from pgm_asc.diagnostics.metrics import posterior_losses


def fit_one(arrays, variant, settings, seed, nodes=61, sigma_bound=3.):
    torch.set_num_threads(1)
    stats = Statistics(arrays)
    target = stats.targets(arrays)[:, -1]
    rng = np.random.default_rng(seed)
    # Consume shared M1 starts in the same order, even for the no-link arms.
    base = {k: float(rng.normal(0, settings["initialization_sd"])) if seed else 0.
            for k in ("log_a", "log_b", "t0", "t1")}
    if "sigma_raw" in KEYS[variant]:
        sd = settings["sigma_starts"][settings["restart_seeds"].index(seed)]
        base["sigma_raw"] = float(sigma_bound*np.arctanh(sd/sigma_bound))
    if "t2" in KEYS[variant]:
        base["t2"] = float(rng.normal(0, settings["initialization_sd"])) if seed else 0.
    initial = {k: base[k] for k in KEYS[variant]}
    params = tensors(initial, True)
    opt = torch.optim.LBFGS(list(params.values()), lr=1., max_iter=settings["max_iter"],
                          tolerance_grad=settings["tolerance_grad"], tolerance_change=settings["tolerance_change"],
                          history_size=20, line_search_fn="strong_wolfe")
    started, calls = time.perf_counter(), 0
    def components():
        joint = -stats.marginal(params, variant, nodes, sigma_bound)[target].mean()/16
        return joint, joint+settings["ridge"]*sum(v.square() for v in params.values())
    def closure():
        nonlocal calls
        calls += 1
        opt.zero_grad()
        loss = components()[-1]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite P1 loss")
        loss.backward()
        return loss
    opt.step(closure)
    opt.zero_grad()
    j, loss = components()
    loss.backward()
    iterations = opt.state[next(iter(params.values()))]["n_iter"]
    return {"params": {k: float(v.detach()) for k,v in params.items()}, "initial": initial,
            "restart_seed": seed, "joint_nll_per_response": float(j.detach()),
            "penalized_loss": float(loss.detach()), "gradient_max_abs": max(float(v.grad.abs()) for v in params.values()),
            "iterations": iterations, "closure_calls": calls,
            "reached_iteration_limit": iterations >= settings["max_iter"],
            "seconds": time.perf_counter()-started, "parameters": len(params), "normal_nodes": nodes if "sigma_raw" in params else 1}


def integration_check(raw, variant, train, dev, coarse, fine, prefixes, sigma_bound):
    """Resolution only: no fitting or model selection on test outcomes."""
    values, predictions, posts = [], [], []
    params = tensors(raw)
    for n in (coarse, fine):
        with torch.no_grad():
            j = float(Statistics(train).joint_loss(params, variant, train, n, sigma_bound))
        p = infer(params, variant, dev, n, sigma_bound)
        _, _, losses = posterior_losses(p.post, dev, prefixes)
        values.append({"train_joint": j, "dev_unique_nll": float(np.nanmean(losses["unique_nll"]))})
        predictions.append(np.array([p.next_answer(i,t) for i in range(len(dev["ids"])) for t in prefixes if t<16]))
        posts.append(p.post[:, np.array(prefixes)-1])
    errors = {k: abs(values[0][k]-values[1][k]) for k in values[0]}
    errors["dev_next_max_abs"] = float(np.max(np.abs(predictions[0]-predictions[1])))
    errors["dev_post_max_abs"] = float(np.max(np.abs(posts[0]-posts[1])))
    return {"coarse_nodes": coarse, "fine_nodes": fine, "coarse": values[0], "fine": values[1], "errors": errors,
            "max_error": max(errors.values())}
