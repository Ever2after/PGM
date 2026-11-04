"""Reference baselines: SC, ASC, ESC and ReASC (offline/online calibration), plus dev-tuned ASC/ReASC.

Semantics follow the official ReASC notebooks (github.com/junseokkim00/ReASC @ 8ee1c76):
exact-string answer grouping, Beta top-two stopping at 0.95, ESC windows of 4, and ReASC
evidence weights max(1, exp(0.7 z)) from bottom-10% windowed self-certainty with a
single-response gate.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betainc
from sklearn.mixture import GaussianMixture

from pgm_asc.data.bank import exact_groups
from pgm_asc.data.features import output_tokens
from pgm_asc.policies import select_candidate

METHODS = ["pass1", "sc16", "asc", "esc", "reasc_offline", "reasc_online", "asc_dev_tuned", "reasc_dev_tuned"]


def offline_calibrate(confidences, correct, target=.9):
    """ReASC offline: z-score statistics and a gate threshold from first responses of the calibration cohort."""
    scores = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    best = None
    best_acc = None
    reached = False
    for threshold in np.sort(scores):
        mask = scores >= threshold
        acc = float(correct[mask].mean())
        if acc >= target:
            best, best_acc, reached = float(threshold), acc, True
            break
        if best_acc is None or acc > best_acc:
            best, best_acc = float(threshold), acc
    mean_correct = float(scores[correct].mean()) if correct.any() else None
    gate = max(best, mean_correct) if mean_correct is not None else float(scores.max())
    std = float(scores.std())
    return {"mean": float(scores.mean()), "std": std if std > 0 else 1., "zero_std_guard": std == 0,
            "gate": gate, "target": target, "target_reached": reached, "no_correct_guard": not correct.any(),
            "lambda": .7, "calibration_questions": len(scores), "fit_score": "lowest_group", "inference_score": "bottom10"}


def online_calibrate(first_scores, target=.9):
    """ReASC online: label-free two-component GMM on the evaluation questions' first responses."""
    scores = np.asarray(first_scores, dtype=np.float64)
    gmm = GaussianMixture(2, covariance_type="full", random_state=0).fit(scores[:, None])
    high = int(np.argmax(gmm.means_.ravel()))
    ordered = np.sort(scores)
    posterior = gmm.predict_proba(ordered[:, None])[:, high]
    accepted = ordered[posterior >= target]
    tau = float(accepted[0]) if len(accepted) else float(ordered[-1])
    std = float(scores.std())
    return {"mean": float(scores.mean()), "std": std if std > 0 else 1., "zero_std_guard": std == 0,
            "gate": max(float(gmm.means_[high, 0]), tau), "target": target, "lambda": .7,
            "calibration_questions": len(scores), "uses_labels": False,
            "gmm": {"means": gmm.means_.ravel().tolist(), "variances": gmm.covariances_.ravel().tolist(),
                    "weights": gmm.weights_.tolist(), "converged": bool(gmm.converged_)}}


def trace(predictions, bottom_scores, method, calibration=None):
    """Chosen candidate and Beta top-two confidence after each response (ASC, or ReASC-weighted)."""
    _, ids = exact_groups(predictions)
    n = len(ids)
    weights = np.zeros(n)
    choices, confidences = [], []
    for i, label in enumerate(ids):
        value = 1.
        if method == "reasc":
            z = (bottom_scores[i] - calibration["mean"]) / calibration["std"]
            value = max(1., float(np.exp(.7 * z)))
            if not np.isfinite(value):
                raise FloatingPointError("ReASC evidence overflow")
        weights[label] += value
        count = int(ids[:i + 1].max()) + 1
        chosen = int(weights[:count].argmax())
        ordered = sorted(weights[:count], reverse=True) + [0.]
        prob = float(1. - betainc(ordered[0] + 1, ordered[1] + 1, .5))
        if method == "reasc" and i == 0 and bottom_scores[0] >= calibration["gate"]:
            prob = 1.
        choices.append(chosen)
        confidences.append(prob)
    return {"chosen": np.asarray(choices), "confidence": np.asarray(confidences), "can_stop": np.ones(n, dtype=bool)}


def decide(predictions, bottom_scores, method, calibration=None):
    """(stop index, chosen candidate, reason) for the notebook-default baselines."""
    _, ids = exact_groups(predictions)
    n = len(ids)
    if method == "pass1":
        return 0, int(ids[0]), "fixed"
    if method == "sc16":
        return n - 1, int(np.bincount(ids).argmax()), "fixed"
    if method == "esc":
        end = n
        for start in range(0, n, 4):
            block = ids[start:start + 4]
            if len(set(block)) == 1:
                end = start + len(block)
                break
        return end - 1, int(np.bincount(ids[:end]).argmax()), "unanimous_block" if end < n else "cap"
    history = trace(predictions, bottom_scores, "reasc" if method.startswith("reasc") else "asc", calibration)
    hits = np.flatnonzero(history["confidence"] >= .95)
    index = int(hits[0]) if len(hits) else n - 1
    return index, int(history["chosen"][index]), "confidence" if len(hits) else "cap"


def tuned_decision(bank, method, calibration, beta):
    """ASC/ReASC with the stopping threshold ``beta`` chosen on dev (None = never stop early)."""
    pred = [r["prediction"] for r in bank.records]
    scores = [r["bottom10_confidence"] for r in bank.records]
    h = trace(pred, scores, "reasc" if method == "reasc" else "asc", calibration)
    if method == "reasc" and scores[0] >= calibration["gate"]:
        return 0, int(h["chosen"][0]), "single_gate"
    hits = np.flatnonzero(h["confidence"] >= beta) if beta is not None else []
    i = int(hits[0]) if len(hits) else 15
    return i, int(h["chosen"][i]), "confidence" if len(hits) else "cap"


def tune_thresholds(banks, calibration, sc, gammas, tolerance):
    """Dev-tuned ASC and ReASC: same selection rule as the learned heads."""
    result = {}
    for method in ["asc", "reasc"]:
        candidates = []
        for gamma in gammas:
            acc, tokens, samples = [], [], []
            for b in banks:
                i, c, _ = tuned_decision(b, method, calibration, gamma)
                acc.append(b.labels["canonical_correct"][c])
                samples.append(i + 1)
                tokens.append(sum(output_tokens(r) for r in b.records[:i + 1]))
            candidates.append({"gamma": gamma, "accuracy": float(np.mean(acc)), "mean_tokens": float(np.mean(tokens)),
                               "mean_samples": float(np.mean(samples))})
        result[method] = {"selected": select_candidate(candidates, sc, tolerance), "candidates": candidates}
    return result


def baseline_decision(bank, method, offline, online, tuned):
    """Decision of any baseline in ``METHODS`` for one question."""
    if method.endswith("_dev_tuned"):
        kind = method.removesuffix("_dev_tuned")
        return tuned_decision(bank, kind, offline, tuned[kind]["selected"]["gamma"])
    pred = [r["prediction"] for r in bank.records]
    scores = [r["bottom10_confidence"] for r in bank.records]
    calibration = offline if method == "reasc_offline" else online if method == "reasc_online" else None
    i, c, why = decide(pred, scores, method, calibration)
    return i, c, ("esc_" + why if method == "esc" else why)


def sc_accuracy(banks):
    """SC16 accuracy (majority over all 16 responses, first-observed tie break)."""
    correct = []
    for b in banks:
        _, ids = exact_groups([r["prediction"] for r in b.records])
        correct.append(b.labels["canonical_correct"][int(np.bincount(ids).argmax())])
    return float(np.mean(correct))
