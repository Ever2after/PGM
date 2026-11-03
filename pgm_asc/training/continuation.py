"""Supervised ExtraTrees continuation fitting and future-payoff targets."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from pgm_asc.core.io import read_json, utc_now, write_json
from pgm_asc.policies.continuation import continuation_features


def continuation_targets(histories, data, metadata, horizon=4, cap=None):
    n = len(histories[0]["chosen"])
    cap = n if cap is None else cap
    if not 1 <= cap <= n:
        raise ValueError("Continuation target cap exceeds the bank")
    gold = data["gold_groups"].numpy()
    target = []
    for qi, history in enumerate(histories):
        for index, chosen in enumerate(history["chosen"]):
            future = min(cap - 1, index + horizon) if index < cap else index
            future_choice = int(history["chosen"][future])
            now = chosen >= 0 and bool(gold[qi * n + index, chosen])
            later = future_choice >= 0 and bool(gold[qi * n + future, future_choice])
            token_delta = metadata[qi * n + future]["token_cost"] - metadata[qi * n + index]["token_cost"]
            target.append([float(later) - float(now), float(token_delta)])
    return np.asarray(target)


def fit_continuation(train_traces, dev_traces, train, dev, train_meta, dev_meta, root, cap=32):
    path = Path(root) / "continuation.joblib"
    resultpath = Path(root) / "continuation_fit.json"
    if path.exists() and resultpath.exists():
        return joblib.load(path), read_json(resultpath)
    tx, dx = continuation_features(train_traces, train), continuation_features(dev_traces, dev)
    ty = continuation_targets(train_traces, train, train_meta, cap=cap)
    dy = continuation_targets(dev_traces, dev, dev_meta, cap=cap)
    # Only positive future horizons train the head; the cap handles t=N.
    train_valid = train["t"].numpy() < cap
    dev_valid = dev["t"].numpy() < cap
    scale = np.array([1., max(float(ty[train_valid, 1].mean()), 1.)])
    candidates = []
    for leaf in [32, 128]:
        model = ExtraTreesRegressor(n_estimators=100, max_depth=8, min_samples_leaf=leaf,
                                   random_state=42, n_jobs=4)
        model.fit(tx[train_valid], ty[train_valid] / scale)
        errors = (model.predict(dx[dev_valid]) - dy[dev_valid] / scale) ** 2
        candidates.append((float(errors.mean()), leaf, model))
    score, leaf, model = min(candidates, key=lambda r: (r[0], -r[1]))
    fitted = {"model": model, "output_scale": scale}
    joblib.dump(fitted, path)
    result = {"created_at": utc_now(), "leaf": leaf, "dev_normalized_mse": score,
              "grid": [{"leaf": c[1], "dev_normalized_mse": c[0]} for c in candidates],
              "train_only_future_targets": True, "horizon": 4, "cap": cap, "input_features": tx.shape[1],
              "corrected_targets": int((ty[train_valid, 0] > 0).sum()),
              "regressed_targets": int((ty[train_valid, 0] < 0).sum())}
    write_json(resultpath, result)
    return fitted, result
