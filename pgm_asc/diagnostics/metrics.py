"""Scores with explicit targets; prefixes are never treated as independent questions."""
from __future__ import annotations

import numpy as np

EPS = 1e-12
GAIN_EDGES = np.array([-1., -.1, -.05, -.02, 0., .01, .02, .05, .1, .2, .5, 1.])


def reference(arrays):
    """Plurality of VALID answers, ties by first observation, without using labels."""
    ids, valid = arrays["ids"].astype(int), arrays["valid"]
    counts = np.cumsum(np.eye(16, dtype=int)[ids] * valid[..., None], axis=1)
    chosen = counts.argmax(2)
    return np.where(counts.sum(2) > 0, chosen, -1)


def labels(arrays, chosen):
    return (chosen >= 0) & np.take_along_axis(arrays["correct"], np.maximum(chosen, 0), axis=1)


def targets(arrays):
    ids, valid = arrays["ids"].astype(int), arrays["valid"]
    seen = np.cumsum(np.eye(16, dtype=int)[ids] * valid[..., None], axis=1) > 0
    accepted = seen & arrays["correct"][:, None, :]
    unseen = ~accepted.any(2)
    # A first-correct convention is ONLY used on the unique-correct subset, or
    # as an explicitly named legacy diagnostic; it is not semantic equivalence.
    canonical = np.where(unseen, 16, accepted.argmax(2))
    unique = accepted[:, -1].sum(1) <= 1
    return accepted, unseen, canonical, unique


def validate_posterior(post, arrays):
    shape = (*arrays["ids"].shape, 17)
    if post.shape != shape or not np.isfinite(post).all() or (post < 0).any():
        raise ValueError(f"Invalid posterior; expected finite nonnegative {shape}")
    if not np.allclose(post.sum(2), 1, atol=2e-6):
        raise ValueError("Posterior is not normalized")
    # Empty extracted answers may have mass in v1; they are observed wrong
    # categories, never OTHER. Only candidates that have not appeared are masked.
    seen = np.cumsum(np.eye(16)[arrays["ids"].astype(int)], axis=1) > 0
    if np.any(post[..., :16][~seen] > 2e-6):
        raise ValueError("Posterior assigns probability to a future candidate")


def binary_losses(prob, truth):
    prob, truth = np.asarray(prob, float), np.asarray(truth, float)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    clipped = np.clip(prob, EPS, 1 - EPS)
    return {"nll": -(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)),
            "brier": (prob - truth) ** 2,
            "bias": prob - truth}


def reliability(prob, truth, edges=None):
    prob, truth = np.asarray(prob).ravel(), np.asarray(truth).ravel()
    edges = np.linspace(0, 1, 11) if edges is None else np.asarray(edges)
    idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        take = idx == k
        if take.any():
            rows.append({"lower": float(edges[k]), "upper": float(edges[k + 1]), "n": int(take.sum()),
                         "prediction": float(prob[take].mean()), "observed": float(truth[take].mean())})
    return rows


def score_binary(prob, truth):
    out = {k: float(v.mean()) for k, v in binary_losses(prob, truth).items()}
    bins = reliability(prob, truth)
    out.update(n=int(np.size(truth)), prediction=float(np.mean(prob)), observed=float(np.mean(truth)),
               ece=sum(r["n"] * abs(r["prediction"] - r["observed"]) for r in bins) / np.size(truth))
    return out, bins


def posterior_losses(post, arrays, prefixes):
    """Per-question losses averaged over the declared, equally weighted prefixes."""
    validate_posterior(post, arrays)
    accepted, unseen, canonical, unique = targets(arrays)
    selected = reference(arrays)
    hit = labels(arrays, selected)
    p_ref = np.where(selected >= 0, np.take_along_axis(post[..., :16], np.maximum(selected, 0)[..., None], 2)[..., 0], 0.)
    mass = (post[..., :16] * accepted).sum(2) + post[..., 16] * unseen
    legacy_p = np.take_along_axis(post, canonical[..., None], axis=2)[..., 0]
    onehot = np.eye(17)[canonical]
    loss = {"accepted_set_nll": -np.log(np.clip(mass, EPS, 1)),
            "unique_nll": np.where(unique[:, None], -np.log(np.clip(legacy_p, EPS, 1)), np.nan),
            "unique_brier": np.where(unique[:, None], ((post - onehot) ** 2).sum(2), np.nan)}
    for name, prob, truth in [("reference", p_ref, hit), ("other", post[..., 16], unseen)]:
        loss.update({f"{name}_{k}": v for k, v in binary_losses(prob, truth).items()})
    rows, bins = [], []
    for t in prefixes:
        j = t - 1
        row = {"prefix": t, "questions": len(post), "unique_questions": int(unique.sum()),
               "reference_accuracy": float(hit[:, j].mean()),
               "model_argmax_accuracy": float(arrays["correct"][np.arange(len(post)), post[:, j, :16].argmax(1)].mean()),
               "legacy_first_correct_nll": float(-np.log(np.clip(legacy_p[:, j], EPS, 1)).mean())}
        for k, v in loss.items():
            row[k] = float(np.nanmean(v[:, j])) if np.isfinite(v[:, j]).any() else None
        for name, prob, truth in [("reference", p_ref[:, j], hit[:, j]), ("other", post[:, j, 16], unseen[:, j])]:
            score, rb = score_binary(prob, truth)
            row.update({f"{name}_{k}": v for k, v in score.items() if k in ("ece", "prediction", "observed")})
            bins += [{"prefix": t, "event": name, **b} for b in rb]
        rows.append(row)
    js = np.array(prefixes) - 1
    per_question = {k: np.mean(v[:, js], axis=1) for k, v in loss.items()}
    return rows, bins, per_question


def bootstrap_mean(values, replicates=1000, seed=20261125):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "ci95": None, "questions": 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates)
    for start in range(0, replicates, 100):
        n = min(100, replicates - start)
        boot[start:start + n] = values[rng.integers(len(values), size=(n, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(), "questions": len(values)}


def future_scores(forecasts, arrays, prefixes, horizons):
    """Forecasts [Q,P,H,...]; horizon targets use suffixes ONLY in the scorer."""
    selected = reference(arrays)
    hit = labels(arrays, selected)
    sample_hit = np.take_along_axis(arrays["correct"], arrays["ids"].astype(int), 1) & arrays["valid"]
    rows, bins = [], []
    values = {}

    def collect(name, value):
        values.setdefault(name, []).append(value)
    for pi, t in enumerate(prefixes):
        for hi, h in enumerate(horizons):
            if t + h > 16:
                continue
            events = {"future_correct": sample_hit[:, t:t+h].any(1),
                      "leader_change": selected[:, t+h-1] != selected[:, t-1],
                      "future_reference_correct": hit[:, t+h-1]}
            if "repair" in forecasts:
                events.update(repair=~hit[:, t-1] & hit[:, t+h-1], damage=hit[:, t-1] & ~hit[:, t+h-1])
            for name, truth in events.items():
                score, rb = score_binary(forecasts[name][:, pi, hi], truth)
                zero_events = ((forecasts[name][:, pi, hi] == 0) & truth) | ((forecasts[name][:, pi, hi] == 1) & ~truth)
                rows.append({"prefix": t, "horizon": h, "event": name, **score,
                             "zero_probability_events": int(zero_events.sum())})
                for metric, value in binary_losses(forecasts[name][:, pi, hi], truth).items():
                    collect(f"{name}_{metric}", value)
            predicted = forecasts["gain"][:, pi, hi]
            actual = hit[:, t+h-1].astype(float) - hit[:, t-1].astype(float)
            mse, bias = (predicted - actual) ** 2, predicted - actual
            collect("gain_mse", mse)
            collect("gain_bias", bias)
            collect("gain_zero_mse", actual ** 2)
            collect("gain_mse_minus_zero", mse - actual ** 2)
            collect("gain_mse_mc_corrected", mse - forecasts["gain_mc_se"][:, pi, hi] ** 2)
            rb = reliability(predicted, actual, GAIN_EDGES)
            rows.append({"prefix": t, "horizon": h, "event": "gain", "n": len(actual),
                         "mse": float(mse.mean()), "bias": float(bias.mean()),
                         "zero_gain_mse": float((actual ** 2).mean()),
                         "prediction": float(predicted.mean()), "observed": float(actual.mean()),
                         "calibration_error": sum(r["n"] * abs(r["prediction"] - r["observed"]) for r in rb) / len(actual),
                         "mean_mc_se": float(forecasts["gain_mc_se"][:, pi, hi].mean())})
            bins += [{"prefix": t, "horizon": h, **b} for b in rb]
        if t < 16:
            obs = arrays["ids"][:, t].astype(int)
            previously_seen = (arrays["ids"][:, :t] == obs[:, None]) & arrays["valid"][:, :t]
            target = np.where(previously_seen.any(1), obs, 16)
            target = np.where(arrays["valid"][:, t], target, 17)
            p = forecasts["next_answer"][:, pi]
            actual_p = p[np.arange(len(p)), target]
            rows.append({"prefix": t, "horizon": 1, "event": "next_answer", "n": len(p),
                         "nll": float(-np.log(np.clip(actual_p, EPS, 1)).mean()),
                         "brier": float(((p - np.eye(18)[target]) ** 2).sum(1).mean()),
                         "zero_probability_events": int((actual_p == 0).sum()),
                         "invalid_observations": int((target == 17).sum())})
            collect("next_answer_nll", -np.log(np.clip(actual_p, EPS, 1)))
            collect("next_answer_brier", ((p - np.eye(18)[target]) ** 2).sum(1))
    per_question = {k: np.mean(v, axis=0) for k, v in values.items()}
    return rows, bins, per_question
