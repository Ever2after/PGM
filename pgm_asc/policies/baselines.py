"""Native baseline traces, confidence calibration, and support mapping."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import betainc, logsumexp

from pgm_asc.core.answers import answer_groups
from pgm_asc.core.io import digest, read_json


def cges_distribution(ids, confidence, candidate_count):
    """Recompute ALL past evidence with CURRENT support size, including OTHER."""
    k = candidate_count + 1
    if candidate_count == 0:
        return np.array([1.])
    ids = np.asarray(ids)
    confidence = np.clip(np.asarray(confidence), 1e-12, 1-1e-12)
    valid = ids >= 0
    ids, confidence = ids[valid], confidence[valid]
    log_other = np.log1p(-confidence) - np.log(k - 1)
    scores = np.full(k, log_other.sum())
    for label in range(candidate_count):
        mask = ids == label
        scores[label] += (np.log(confidence[mask]) - log_other[mask]).sum()
    return np.exp(scores - logsumexp(scores))


def trace(answers, features, method, settings=None):
    """Causal prefix decisions. Returned choices use first-observed answer IDs."""
    settings = settings or {}
    _, ids = answer_groups(answers)
    chosen, confidence, can_stop, distributions = [], [], [], []
    weights = np.zeros(len(answers), dtype=float)
    for i, label in enumerate(ids):
        t = i + 1
        count = max(ids[:t], default=-1) + 1
        valid = label >= 0
        if valid:
            weight = 1.
            if method == "reasc":
                score = features[i]["reasc_bottom10_group128"]
                z = (score - settings["mean"]) / settings["std"]
                weight = max(1., float(np.exp(settings.get("lambda", .7) * z)))
            weights[label] += weight
        leader = int(np.argmax(weights[:count])) if count else -1
        value, eligible, distribution = 0., count > 0, None
        if method in ("asc", "reasc") and count:
            top = sorted(weights[:count], reverse=True) + [0.]
            value = float(1 - betainc(top[0] + 1, top[1] + 1, .5))
            if method == "reasc" and t == 1 and settings.get("gate") is not None:
                if valid and features[0]["reasc_bottom10_group128"] >= settings["gate"]:
                    value = 1.  # Stage-1 decision flag, NOT a joint answer posterior.
        elif method == "esc":
            window = settings.get("window", 4)
            # Original Algorithm 1: disjoint blocks, and a unanimous block
            # replaces previous votes on early stop. Not a rolling window.
            block = ids[max(0, t - window):t]
            unanimous = t % window == 0 and len(block) == window and block[0] >= 0 and len(set(block)) == 1
            value = float(unanimous)
            if unanimous:
                leader = block[0]
        elif method == "cges":
            distribution = cges_distribution(ids[:t], [f[settings["confidence"]] for f in features[:t]], count)
            if count:
                leader = int(np.argmax(distribution[:count]))
                value = float(distribution[leader])
                eligible = distribution[-1] < distribution.max()
        elif method == "sc":
            eligible = False
        chosen.append(leader)
        confidence.append(value)
        can_stop.append(eligible)
        distributions.append(distribution)
    result = {"chosen": np.asarray(chosen), "confidence": np.asarray(confidence),
              "can_stop": np.asarray(can_stop), "distributions": distributions}
    if method == "esc":
        result["budget_chosen"] = int(np.argmax(weights[:count])) if count else -1
    return result


def read_split(config, split, metadata):
    root = Path(config["run_dir"]) / split
    n = config["samples_per_question"]
    question_ids = [r["question_id"] for r in metadata[::n]]
    banks = []
    for qid in question_ids:
        answers, features = [], []
        for i in range(n):
            name = f"{qid}__s{i:03d}.json"
            response = read_json(root / "scored_responses" / name)
            feature = read_json(root / "baseline_features" / name)
            if digest(response["token_ids"]) != feature["source_token_hash"]:
                raise ValueError("Supplemental confidence token hash mismatch")
            answers.append(response["answer"])
            features.append(feature)
        banks.append({"answers": answers, "features": features, "question_id": qid})
    return banks


def traces(banks, method, settings=None):
    return [trace(b["answers"], b["features"], method, settings) for b in banks]


def replay(traces_, data, metadata, n, gamma):
    gold = data["gold_groups"].numpy()
    rows, selected, correct, samples, costs, recoverable = [], [], [], [], [], []
    for qi, tr in enumerate(traces_):
        passing = np.flatnonzero(tr["can_stop"] & (tr["confidence"] >= gamma)) if gamma is not None else []
        t0 = int(passing[0]) if len(passing) else n - 1
        offset = qi * n
        label = int(tr.get("budget_chosen", tr["chosen"][-1])) if gamma is None else int(tr["chosen"][t0])
        ok = bool(label >= 0 and gold[offset + t0, label])
        final_label = int(tr.get("budget_chosen", tr["chosen"][-1]))
        final_ok = bool(final_label >= 0 and gold[offset + n - 1, final_label])
        correct.append(ok)
        samples.append(t0 + 1)
        costs.append(metadata[offset + t0]["token_cost"])
        recoverable.append(t0 < n - 1 and not ok and final_ok)
        selected.append(offset + t0)
        rows.append({"question_id": metadata[offset]["question_id"], "t": t0 + 1,
                     "chosen": label, "correct": ok, "token_cost": costs[-1],
                     "recoverable_error": recoverable[-1]})
    return {"accuracy": float(np.mean(correct)), "mean_tokens": float(np.mean(costs)),
            "mean_samples": float(np.mean(samples)), "recoverable_errors": int(sum(recoverable)),
            "early_stop_rate": float(np.mean(np.asarray(samples) < n)), "rows": rows}


def reasc_calibrate(banks, data, n):
    first = np.asarray([b["features"][0]["reasc_bottom10_group128"] for b in banks])
    ids = [answer_groups(b["answers"])[1][0] for b in banks]
    gold = data["gold_groups"].numpy()
    correct = np.asarray([label >= 0 and gold[qi * n, label] for qi, label in enumerate(ids)])
    gates = [t for t in sorted(set(first)) if correct[first >= t].mean() >= .9]
    gate = max(float(first[correct].mean()), float(gates[0])) if gates and correct.any() else None
    return {"mean": float(first.mean()), "std": max(float(first.std()), 1e-6),
            "gate": gate, "lambda": .7, "target_gate_accuracy": .9,
            "calibration_questions": len(banks), "confidence_scope": "first_response_per_question"}


def cges_nll(traces_, data, n):
    losses = []
    target = data["target"].numpy()
    for qi, tr in enumerate(traces_):
        for t in [8, 16, 32]:
            p = tr["distributions"][t - 1]
            label = int(target[qi * n + t - 1])
            index = len(p) - 1 if label == n else label
            losses.append(-np.log(max(p[index], 1e-300)))
    return float(np.mean(losses))


def cges_logp(histories, n=32):
    """Map changing observed-answer support to stable first-appearance columns."""
    output = np.full((len(histories) * n, n + 1), -1e9, dtype=float)
    for qi, history in enumerate(histories):
        for t0, distribution in enumerate(history["distributions"]):
            distribution = np.asarray(distribution)
            observed = len(distribution) - 1
            output[qi * n + t0, :observed] = np.log(np.maximum(distribution[:-1], 1e-300))
            output[qi * n + t0, n] = np.log(max(float(distribution[-1]), 1e-300))
    return output
