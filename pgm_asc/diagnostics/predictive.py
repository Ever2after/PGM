"""Native count-CRP forecasts under a shared, label-free plurality decision.

Unlike the stopping VOI implementation, future answer choice never depends on a
model's tie-break or on recomputing its posterior. This isolates the predictive
distribution. Future confidence integrates out for this fixed decision rule;
copy arms require an unspecified future-similarity distribution. The flat-wrong
likelihood is unnormalized.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.special import logsumexp


def capability(spec):
    if spec.wrong != "crp":
        return "unavailable: flat-wrong likelihood does not define a normalized answer generator"
    if spec.copy != "none":
        return "unavailable: no generative distribution for future reasoning similarities"
    if not spec.other:
        return "unavailable in this adapter: OTHER-removed model has no unseen correct-answer support"
    return "native_count_crp"


def joint_prob(log_joint, allowed):
    value = np.where(allowed[:, None], log_joint, -np.inf)
    return np.exp(value - logsumexp(value))


def next_answer_distribution(joint, counts, q, theta):
    """Exact probability of existing slots 0..15, NEW=16, INVALID=17."""
    hyp_counts = np.r_[counts, 0.]
    wrong_total = counts.sum() - hyp_counts
    denom = wrong_total[:, None] + theta[None, :]
    factor = ((1 - q)[None, :] * joint / denom).sum(1)
    correct = (q[None, :] * joint).sum(1)
    out = np.zeros(18)
    out[:16] = counts * (factor.sum() - factor[:16]) + correct[:16]
    out[16] = (((1 - q) * theta)[None, :] * joint / denom).sum() + correct[16]
    if not np.isclose(out.sum(), 1., atol=1e-8):
        raise ValueError("Predictive answer distribution is not normalized")
    return np.maximum(out, 0.)


def simulate(joint, counts0, first0, q, theta, horizons, rng, simulations, transition_details=False, *, latent_draws=None):
    """Draw complete answer continuations. Latent true answer is known per draw.

    Labels 0..15 are existing slots, 16 an unseen correct answer, and 17..31
    newly discovered wrong answers. Appearance times, NOT numeric labels, break
    ties: an unseen correct answer can arrive after a new wrong answer.
    """
    if latent_draws is None:
        draws = rng.choice(joint.size, simulations, p=joint.ravel())
        c, grid = draws // joint.shape[1], draws % joint.shape[1]
        qs, th = q[grid], theta[grid]
    else:
        # A hierarchical model samples these jointly once per continuation.
        c, qs, th = (np.asarray(x) for x in latent_draws)
        if any(x.shape != (simulations,) for x in (c, qs, th)):
            raise ValueError("Latent draws must have one value per continuation")
        if np.any((c < 0) | (c > 16)) or np.any((qs <= 0) | (qs >= 1)) or np.any(th <= 0):
            raise ValueError("Invalid latent continuation draws")
    rows = np.arange(simulations)
    counts = np.zeros((simulations, 32))
    counts[:, :16] = counts0
    first = np.full((simulations, 32), 1000, int)
    first[:, :16] = first0
    nxt = np.full(simulations, 17)
    any_correct = np.zeros(simulations, bool)
    leader0 = int(np.where(counts0 == counts0.max(), first0, 1000).argmin()) if counts0.sum() else -1
    hit0 = c == leader0
    result = {}
    for step in range(1, max(horizons) + 1):
        correct = rng.random(simulations) < qs
        wrong = counts.copy()
        wrong[rows, c] = 0
        total = wrong.sum(1)
        new = rng.random(simulations) < th / (total + th)
        u = rng.random(simulations) * np.maximum(total, 1)
        old = (np.cumsum(wrong, axis=1) > u[:, None]).argmax(1)
        lab = np.where(correct, c, np.where(new | (total == 0), nxt, old))
        nxt += ~correct & (new | (total == 0))
        first[rows, lab] = np.minimum(first[rows, lab], step)
        counts[rows, lab] += 1
        any_correct |= correct
        if step in horizons:
            chosen = np.where(counts == counts.max(1, keepdims=True), first, 1000).argmin(1)
            hit = chosen == c
            result[step] = {"future_correct": float(any_correct.mean()),
                            "leader_change": float((chosen != leader0).mean()),
                            "future_reference_correct": float(hit.mean()),
                            "gain_mc_se": float(hit.std(ddof=1) / np.sqrt(simulations))}
            if transition_details:
                gain = hit.astype(float) - hit0
                repair, damage = hit & ~hit0, ~hit & hit0
                result[step].update(repair=float(repair.mean()), damage=float(damage.mean()),
                                    repair_mc_se=float(repair.std(ddof=1) / np.sqrt(simulations)),
                                    damage_mc_se=float(damage.std(ddof=1) / np.sqrt(simulations)),
                                    current_correct_mc=float(hit0.mean()),
                                    gain_paired=float(gain.mean()),
                                    gain_paired_mc_se=float(gain.std(ddof=1) / np.sqrt(simulations)))
    return result


def forecast(arrays, posterior, q, theta, prefixes, horizons, simulations=256, seed=20261125, key="", *,
             paired_gain=False, transition_details=False):
    """Only ids/valid and model posterior enter; correctness arrays are not read."""
    if simulations < 2:
        raise ValueError("At least two Monte Carlo simulations are required")
    nq = len(arrays["ids"])
    shape = (nq, len(prefixes), len(horizons))
    result = {k: np.full(shape, np.nan) for k in ("future_correct", "leader_change", "future_reference_correct", "gain", "gain_mc_se")}
    if transition_details:
        result.update({k: np.full(shape, np.nan) for k in ("repair", "damage", "repair_mc_se", "damage_mc_se", "current_correct_mc")})
    result["next_answer"] = np.full((nq, len(prefixes), 18), np.nan)
    for pi, t in enumerate(prefixes):
        hs = [h for h in horizons if t + h <= 16]
        if t == 16:
            continue
        for i in range(nq):
            ids, valid = arrays["ids"][i, :t].astype(int), arrays["valid"][i, :t]
            counts = np.bincount(ids[valid], minlength=16).astype(float)
            first = np.full(16, 1000)
            for j in range(t):
                if valid[j]:
                    first[ids[j]] = min(first[ids[j]], j - t)
            joint = joint_prob(posterior.joint[i, t-1], posterior.allowed[i, t-1])
            result["next_answer"][i, pi] = next_answer_distribution(joint, counts, q, theta)
            if not hs:
                continue
            identity = f"{seed}/{key}/{arrays['question_id'][i]}/{t}".encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity).digest()[:8], "little"))
            samples = simulate(joint, counts, first, q, theta, hs, rng, simulations,
                               transition_details=paired_gain or transition_details)
            leader = int(np.where(counts == counts.max(), first, 1000).argmin()) if counts.sum() else -1
            now = joint[leader].sum() if leader >= 0 else 0.
            for hi, h in enumerate(horizons):
                if h not in samples:
                    continue
                for name in ("future_correct", "leader_change", "future_reference_correct"):
                    result[name][i, pi, hi] = samples[h][name]
                result["gain"][i, pi, hi] = samples[h]["gain_paired"] if paired_gain else samples[h]["future_reference_correct"] - now
                result["gain_mc_se"][i, pi, hi] = samples[h]["gain_paired_mc_se"] if paired_gain else samples[h]["gain_mc_se"]
                if transition_details:
                    for name in ("repair", "damage", "repair_mc_se", "damage_mc_se", "current_correct_mc"):
                        result[name][i, pi, hi] = samples[h][name]
    return result
