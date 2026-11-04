"""Training one head: weighted prefix NLL with early stopping on dev."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from pgm_asc.model import batch, make, predict
from pgm_asc.utils import log

# Prefixes 1-2, 3-4, 5-8 and 9-16 each receive total weight 0.4.
PREFIX_WEIGHTS = [.2, .2] + [.1] * 2 + [.05] * 4 + [.025] * 8


def fit(train, dev, spec, params, seed, label=""):
    """Train until dev NLL stops improving for ``patience`` epochs.

    Returns (model at the best dev epoch, checkpoint entry, summary, per-epoch history).
    ``params``: epochs, patience, lr, weight_decay, question_batch.
    """
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    dim = train["embeddings"].shape[-1]
    model = make(spec, dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    weights = torch.tensor(PREFIX_WEIGHTS)
    rng = np.random.default_rng(seed)
    nq = len(train["t"]) // 16
    best, bad, history, active, started = float("inf"), 0, [], None, time.perf_counter()
    for epoch in range(params["epochs"]):
        model.train()
        order = rng.permutation(nq)
        losses = []
        for start in range(0, nq, params["question_batch"]):
            idx = torch.from_numpy((order[start:start + params["question_batch"], None] * 16 + np.arange(16)).reshape(-1))
            lp = model(batch(train, idx, model))
            loss = (F.nll_loss(lp, train["target"][idx], reduction="none").reshape(-1, 16) * weights).sum(1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            optimizer.zero_grad()
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise FloatingPointError("Nonfinite gradient")
            if active is None:
                active = sum(int((v.grad != 0).sum()) for v in model.parameters() if v.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        lp = predict(model, dev)
        target = dev["target"].numpy()
        score = float((-lp[np.arange(len(lp)), target].reshape(-1, 16) * weights.numpy()).sum(1).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "dev_nll": score})
        if score < best - 1e-7:
            best, bad, best_epoch, state = score, 0, epoch + 1, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if epoch == 0 or (epoch + 1) % 20 == 0:
            log("fit", head=label, seed=seed, epoch=epoch + 1, dev_nll=score)
        if bad >= params["patience"]:
            break
    model.load_state_dict(state)
    summary = {"seed": seed, "spec": spec, "epochs": len(history), "best_epoch": best_epoch, "dev_nll": best,
               "alpha": float(model.alpha.detach()), "parameters": sum(v.numel() for v in model.parameters()),
               "nonzero_gradient_entries_first_step": active, "seconds": time.perf_counter() - started}
    return model, {"state_dict": state, "spec": spec, "embedding_dim": dim, "seed": seed}, summary, history
