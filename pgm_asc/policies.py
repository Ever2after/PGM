"""Calibration, stopping rules and threshold selection for the learned heads."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


# ---------------------------------------------------------------- calibration
def temperature(logp, target):
    """Global temperature T minimizing NLL of the first correct candidate (or OTHER)."""
    def loss(log_t):
        scaled = logp / np.exp(log_t)
        scaled -= logsumexp(scaled, axis=1, keepdims=True)
        return float(-scaled[np.arange(len(target)), target].mean())
    fitted = minimize_scalar(loss, bounds=(-4, 4), method="bounded")
    if not fitted.success:
        raise RuntimeError("Temperature optimization failed")
    return {"T": float(np.exp(fitted.x)), "nll": float(fitted.fun), "before_nll": loss(0), "log_T_bounds": [-4, 4]}


def global_calibrator(lp, data):
    fitted = temperature(np.asarray(lp, dtype=np.float64), data["target"].numpy())
    return {"kind": "global", "T": fitted["T"], "fit": fitted}


def calibrate(lp, calibrator):
    lp = np.asarray(lp, dtype=np.float64)
    if calibrator["kind"] != "global":
        raise ValueError("Only global temperature calibration is used")
    result = lp / calibrator["T"]
    return result - logsumexp(result, axis=1, keepdims=True)


# ---------------------------------------------------------------- stopping
def traces(logp, data):
    """Per question: chosen candidate, its probability, OTHER probability, and whether stopping is allowed."""
    if len(logp) % 16:
        raise ValueError("Sixteen prefixes required")
    c = logp.shape[1] - 1
    chosen = logp[:, :c].argmax(1)
    valid = data["candidate_mask"][:, :c].any(1).numpy()
    result = []
    for start in range(0, len(logp), 16):
        s = slice(start, start + 16)
        result.append({"chosen": np.where(valid[s], chosen[s], -1),
                       "confidence": np.exp(logp[np.arange(start, start + 16), chosen[s]]),
                       "can_stop": valid[s] & (logp[s, c] < logp[s, :c].max(1)),
                       "other": np.exp(logp[s, c])})
    return result


def stopping_index(history, policy):
    """First prefix >= minimum whose chosen candidate has p >= gamma and beats OTHER, for ``patience`` prefixes."""
    minimum, patience, cap = (policy[k] for k in ["minimum", "patience", "cap"])
    n = len(history["chosen"])
    if not 1 <= minimum <= cap <= n or patience < 1:
        raise ValueError("Invalid minimum/patience/cap")
    gamma = policy["gamma"]
    chosen = np.asarray(history["chosen"])
    qualifying = np.asarray(history["can_stop"], bool) & (
        np.asarray(history["confidence"]) >= gamma if gamma is not None else False)
    for index in range(minimum - 1, cap):
        lo = index - patience + 1
        if lo >= 0 and qualifying[lo:index + 1].all() and (chosen[lo:index + 1] == chosen[index]).all():
            return index, "confidence"
    return cap - 1, "cap"


def replay_policy(histories, data, metadata, policy):
    """Accuracy and mean consumed output tokens of one stopping policy on precomputed traces."""
    n = len(histories[0]["chosen"])
    gold = data["gold_groups"].numpy()
    rows = []
    for qi, history in enumerate(histories):
        index, reason = stopping_index(history, policy)
        chosen = int(history["chosen"][index])
        correct = chosen >= 0 and bool(gold[qi * n + index, chosen])
        rows.append({"question_id": metadata[qi * n]["question_id"], "t": index + 1, "chosen": chosen,
                     "correct": bool(correct), "reason": reason, "token_cost": metadata[qi * n + index]["token_cost"]})
    return {"accuracy": float(np.mean([r["correct"] for r in rows])),
            "mean_tokens": float(np.mean([r["token_cost"] for r in rows])),
            "mean_samples": float(np.mean([r["t"] for r in rows])), "rows": rows}


def select_candidate(candidates, sc, tolerance=.01):
    """Fewest dev tokens subject to dev accuracy >= SC16 - tolerance; else the most accurate."""
    for row in candidates:
        row["feasible"] = row["accuracy"] >= sc - tolerance - 1e-12
    eligible = [r for r in candidates if r["feasible"]]
    if eligible:
        return min(eligible, key=lambda r: (r["mean_tokens"], -r["accuracy"], r["mean_samples"]))
    return min(candidates, key=lambda r: (-r["accuracy"], r["mean_tokens"], r["mean_samples"]))
