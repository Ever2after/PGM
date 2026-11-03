"""Global, prefix-bucket, and state temperature calibration."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import KFold

from pgm_asc.data.prefixes import bucket_ids, prefix_weights


def temperature(logp, data, majority_metadata=None):
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    target = data["target"].numpy()
    if majority_metadata is not None:
        leader = np.array([r["leader"] for r in majority_metadata])
        valid = data["candidate_mask"][:, :-1].any(1).numpy()
        binary_target = data["gold_groups"].numpy()[np.arange(len(target)), leader] & valid
    def loss(log_t):
        scaled = logp / np.exp(log_t)
        scaled -= logsumexp(scaled, axis=1, keepdims=True)
        if majority_metadata is None:
            return float(-scaled[np.arange(len(target)), target].mean())
        p = np.exp(scaled[np.arange(len(target)), leader])
        p[~valid] = 0
        p = np.clip(p, 1e-12, 1-1e-12)
        return float(-(binary_target * np.log(p) + (~binary_target) * np.log1p(-p)).mean())
    fitted = minimize_scalar(loss, bounds=(-4, 4), method="bounded")
    if not fitted.success:
        raise RuntimeError("Temperature optimization failed")
    return {"T": float(np.exp(fitted.x)), "nll": float(fitted.fun),
            "before_nll": loss(0), "log_T_bounds": [-4, 4]}


def global_calibrator(lp, data):
    fitted = temperature(np.asarray(lp, dtype=np.float64), data)
    return {"kind": "global", "T": fitted["T"], "fit": fitted}


def calibrate(lp, times, parameters):
    lp = np.asarray(lp, dtype=np.float64)
    if parameters["kind"] == "global":
        result = lp / parameters["T"]
    else:
        buckets = bucket_ids(times)
        result = lp.copy()
        result[:, -1] += np.asarray(parameters["bias"])[buckets]
        result /= np.exp(np.asarray(parameters["log_T"])[buckets, None])
    return result - logsumexp(result, axis=1, keepdims=True)


def _fit_bucket(lp, targets, times, global_T, strength):
    # Optimize in float64: float32 in-place division rounds the objective while
    # L-BFGS-B follows a higher-precision gradient, breaking its line search.
    lp = np.asarray(lp, dtype=np.float64)
    ids = bucket_ids(times)
    weights = prefix_weights(times)
    anchor = np.log(global_T)
    initial = np.r_[np.full(6, anchor), np.zeros(6)]

    def objective(theta):
        scaled = lp.copy()
        scaled[:, -1] += theta[6:][ids]
        scaled /= np.exp(theta[:6][ids, None])
        p = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
        nll = logsumexp(scaled, axis=1) - scaled[np.arange(len(lp)), targets]
        residual = p.copy()
        residual[np.arange(len(lp)), targets] -= 1
        grad_t = -(residual * scaled).sum(1) * weights
        grad_b = residual[:, -1] / np.exp(theta[:6][ids]) * weights
        gradient = np.r_[np.bincount(ids, grad_t, minlength=6),
                         np.bincount(ids, grad_b, minlength=6)]
        penalty = strength * (np.mean((theta[:6] - anchor) ** 2) + np.mean(theta[6:] ** 2))
        gradient[:6] += strength * (theta[:6] - anchor) / 3
        gradient[6:] += strength * theta[6:] / 3
        return float(weights @ nll + penalty), gradient

    fitted = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(-4, 4)] * 6 + [(-4, 4)] * 6,
                      options={"maxiter": 200, "ftol": 1e-10})
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError(f"Bucket calibration failed: {fitted.message}")
    return {"kind": "bucket", "log_T": fitted.x[:6].tolist(), "bias": fitted.x[6:].tolist(),
            "strength": strength, "anchor_T": global_T}


def bucket_calibrator(lp, data):
    times = data["t"].numpy()
    target = data["target"].numpy()
    qidx = data["question_index"].numpy()
    questions = np.unique(qidx)
    scores = []
    for strength in [.01, .1, 1.]:
        oof = np.empty_like(lp, dtype=float)
        for train_q, valid_q in KFold(3, shuffle=True, random_state=42).split(questions):
            train = np.isin(qidx, questions[train_q])
            valid = np.isin(qidx, questions[valid_q])
            # Even the shrinkage anchor is fitted without the validation questions.
            anchor = global_calibrator(lp[train], {"target": data["target"][train]})
            parameters = _fit_bucket(lp[train], target[train], times[train], anchor["T"], strength)
            oof[valid] = calibrate(lp[valid], times[valid], parameters)
        loss = float(prefix_weights(times) @ -oof[np.arange(len(target)), target])
        scores.append({"strength": strength, "oof_bucket_nll": loss})
    best = min(scores, key=lambda r: (r["oof_bucket_nll"], -r["strength"]))
    anchor = global_calibrator(lp, data)
    result = _fit_bucket(lp, target, times, anchor["T"], best["strength"])
    result.update(oof_scores=scores, selected_without_calibration_B=True)
    return result


def calibrate_state(lp,temperature):
    a=np.asarray(lp,dtype=float)/temperature
    return a-logsumexp(a,axis=-1,keepdims=True)


def fit_state_temperature(lp,target):
    lp=np.asarray(lp,dtype=float)
    weights=np.array([0.,1/3,1/6,1/6,1/12,1/12,1/12,1/12])
    def loss(logt):
        a=calibrate_state(lp,np.exp(logt));nll=-np.take_along_axis(a,target[:,:,None],-1).squeeze(-1)
        return float((nll*weights).sum(1).mean())
    fit=minimize_scalar(loss,bounds=(-4,4),method='bounded')
    if not fit.success or not np.isfinite(fit.fun):raise ValueError('Temperature fitting failed')
    return {'T':float(np.exp(fit.x)),'nll':float(fit.fun),'before':loss(0.)}
