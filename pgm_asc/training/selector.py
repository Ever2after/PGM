"""Selector fitting with original and prefix-balanced objectives."""
from __future__ import annotations

import copy

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pgm_asc.core.io import log, read_json, utc_now, write_json, write_jsonl
from pgm_asc.data.prefixes import prefix_weights
from pgm_asc.evaluation.metrics import measurements
from pgm_asc.models.selector import PGM, batch_inputs, predict


def fit(train, dev, spec, n, root, seed=42, lr=.001, weight_decay=.0001, epochs=200):
    torch.manual_seed(seed)
    model = PGM(n, **spec).to(train["x"].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    qcount = int(train["question_index"].max()) + 1
    best, best_state, bad = float("inf"), None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        for start in range(0, qcount, 32):
            # Shuffle questions once per epoch; every prefix of a question stays together.
            if start == 0:
                order = rng.permutation(qcount)
            questions = order[start:start + 32]
            indices = torch.from_numpy((questions[:, None] * n + np.arange(n)).reshape(-1)).to(train["x"].device)
            batch = batch_inputs(train, indices, n, model.embedding_dim > 0)
            lp = model(batch)
            loss = F.nll_loss(lp, train["target"][indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PGM loss")
            optimizer.zero_grad()
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite PGM gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        logp, weights = predict(model, dev)
        metrics, *_ = measurements(logp, dev, n)
        # Early stopping follows the plan's ALL-prefix criterion. Final trial
        # selection uses the separately prespecified t=8/16/32 mean NLL.
        criterion = metrics["all_prefix_nll"]
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "dev_all_prefix_nll": criterion, "dev_selection_nll": metrics["selection_nll"],
                        "alpha": float(model.alpha.detach())})
        if criterion < best - 1e-7:
            best, bad, best_state = criterion, 0, copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        else:
            bad += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log("PGM training", spec=spec, seed=seed, lr=lr, epoch=epoch + 1,
                dev_nll=criterion, alpha=float(model.alpha.detach()), elapsed_seconds=time.perf_counter() - started)
        if bad >= 20:
            break
    model.load_state_dict(best_state)
    logp, weights = predict(model, dev)
    metrics, *_ = measurements(logp, dev, n)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    portable = {k: v.cpu() for k, v in best_state.items()}
    torch.save({"state_dict": portable, "spec": spec, "n": n, "seed": seed}, root / "checkpoint.pt")
    np.savez_compressed(root / "dev_predictions.npz", logp=logp, gate_weights=weights)
    write_jsonl(root / "training_history.jsonl", history)
    result = {"spec": spec, "n": n, "seed": seed, "lr": lr, "weight_decay": weight_decay,
              "epochs": len(history), "best_epoch": best_epoch, "alpha": float(model.alpha.detach()),
              "elapsed_seconds": time.perf_counter() - started, "metrics": metrics,
              "gate_mean_weights": weights.mean(0).tolist(),
              "gate_entropy": float(-(weights * np.log(np.maximum(weights, 1e-30))).sum(1).mean()),
              "total_parameter_count": sum(p.numel() for p in model.parameters())}
    write_json(root / "result.json", result)
    return result


def train_bucket_model(train, dev, spec, root, lr, weight_decay, epochs=120, device="cpu"):
    """Same seed/init/architecture/optimizer; only prefix objective changes."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "result.json").exists():
        return read_json(root / "result.json")
    train = {k: v.to(device) for k, v in train.items() if k != "embeddings"}
    dev = {k: v.to(device) for k, v in dev.items() if k != "embeddings"}
    torch.manual_seed(42)
    model = PGM(32, **spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = torch.tensor(prefix_weights(np.arange(1, 33)), dtype=torch.float32, device=device)
    rng = np.random.default_rng(42)
    qcount = int(train["question_index"].max()) + 1
    best, bad, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(qcount)
        losses = []
        for start in range(0, qcount, 32):
            questions = order[start:start + 32]
            indices = torch.from_numpy((questions[:, None] * 32 + np.arange(32)).reshape(-1)).to(device)
            lp = model(batch_inputs(train, indices, 32))
            loss = (F.nll_loss(lp, train["target"][indices], reduction="none").reshape(-1, 32) * weights).sum(1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite weighted training loss")
            optimizer.zero_grad()
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite weighted training gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        lp, _ = predict(model, dev)
        target = dev["target"].cpu().numpy()
        criterion = float((-lp[np.arange(len(lp)), target].reshape(-1, 32) * weights.cpu().numpy()).sum(1).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "dev_bucket_nll": criterion})
        if criterion < best - 1e-7:
            best, bad, best_epoch = criterion, 0, epoch + 1
            state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            log("v2 prefix-weighted training", path=str(root), epoch=epoch + 1, dev_bucket_nll=criterion)
            write_json(root / "progress.json", {"epoch": epoch + 1, "best_epoch": best_epoch, "updated_at": utc_now()})
        if bad >= 20:
            break
    torch.save({"state_dict": {k: v.cpu() for k, v in state.items()}, "spec": spec, "n": 32, "seed": 42}, root / "checkpoint.pt")
    write_jsonl(root / "history.jsonl", history)
    result = {"created_at": utc_now(), "best_epoch": best_epoch, "epochs": epoch + 1,
              "dev_bucket_nll": best, "elapsed_seconds": time.perf_counter() - started,
              "objective": "six_equal_total_mass_prefix_buckets", "seed": 42, "spec": spec,
              "lr": lr, "weight_decay": weight_decay}
    write_json(root / "result.json", result)
    return result
