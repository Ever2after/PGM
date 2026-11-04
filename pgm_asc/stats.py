"""Per-question decision rows, aggregate metrics, paired bootstrap contrasts and Holm adjustment."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from pgm_asc.data.bank import exact_groups

PHASES = ("extraction_inline", "extraction_final")


def decision_row(bank, index, chosen, reason, uses_features, rounded_parameters):
    """Outcome and costs of stopping after ``index + 1`` responses and answering candidate ``chosen``."""
    reps, ids = exact_groups([r["prediction"] for r in bank.records])
    t = index + 1
    consumed = bank.records[:t]
    if chosen < 0 or chosen >= len(reps):
        raise ValueError("Selected unseen answer")
    labels = bank.labels
    # Reference metric: majority of raw-response verdicts for the chosen candidate
    # (ESC uses the full bank, as in the released notebook aggregation).
    verdict_end = 16 if reason.startswith("esc_") else t
    raw = [labels["raw_response_correct"][i] for i in range(verdict_end) if ids[i] == chosen]
    chat = sum(r["reference_chat_tokens"] for r in consumed)
    gen = sum(r["generated_token_count"] for r in consumed)
    extraction_in = sum(r[f"{k}_input_tokens"] for r in consumed for k in PHASES)
    extraction_out = sum(r[f"{k}_output_tokens"] for r in consumed for k in PHASES)
    tf = sum(r["feature_input_tokens"] for r in consumed) if uses_features else 0
    n = rounded_parameters
    return {"question_id": bank.question["question_id"], "answer": reps[chosen], "chosen": chosen, "samples": t,
            "reason": reason, "correct": bool(labels["canonical_correct"][chosen]), "reference_correct": bool(sum(raw) * 2 >= len(raw)),
            "reference_chat_tokens": chat, "generated_tokens": gen, "extraction_input_tokens": extraction_in,
            "extraction_output_tokens": extraction_out, "feature_input_tokens": tf,
            "feature_seconds": sum(r["feature_seconds_share"] for r in consumed) if uses_features else 0.,
            "native_generation_seconds": sum(r["native_generation_seconds_share"] for r in consumed),
            "extraction_seconds": sum(r[f"{k}_seconds_share"] for r in consumed for k in PHASES),
            "output_tokens": gen + extraction_out,
            "reference_proxy_T": n * chat / 1e12,
            "dense_2N_approx_T": 2 * n * (chat + extraction_in + extraction_out + tf) / 1e12}


METRIC_KEYS = ["correct", "reference_correct", "samples", "reference_chat_tokens", "generated_tokens", "extraction_input_tokens",
               "extraction_output_tokens", "feature_input_tokens", "feature_seconds", "native_generation_seconds",
               "extraction_seconds", "reference_proxy_T", "dense_2N_approx_T"]


def metrics(rows):
    ci = binomtest(sum(r["correct"] for r in rows), len(rows)).proportion_ci(confidence_level=.95, method="exact")
    return {"questions": len(rows), "accuracy_ci95_exact": [float(ci.low), float(ci.high)],
            **{k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS},
            "mean_output_tokens": float(np.mean([r["output_tokens"] for r in rows])),
            "total_output_tokens": int(sum(r["output_tokens"] for r in rows)),
            "total_body_tokens": int(sum(r["generated_tokens"] for r in rows))}


def paired(a, b, seed=20261102):
    """Question-paired accuracy difference (a - b), output-token ratio, bootstrap CIs and exact McNemar p."""
    if [r["question_id"] for r in a] != [r["question_id"] for r in b]:
        raise ValueError("Unpaired cohorts")
    aa = np.array([r["correct"] for r in a], int)
    bb = np.array([r["correct"] for r in b], int)
    ac = np.array([r["output_tokens"] for r in a])
    bc = np.array([r["output_tokens"] for r in b])
    ad = np.array([r["dense_2N_approx_T"] for r in a])
    bd = np.array([r["dense_2N_approx_T"] for r in b])
    rng = np.random.default_rng(seed)
    delta, ratio, dense = [], [], []
    for _ in range(10):
        idx = rng.integers(0, len(a), (1000, len(a)))
        delta.extend((aa[idx] - bb[idx]).mean(1))
        ratio.extend(ac[idx].mean(1) / bc[idx].mean(1))
        dense.extend(ad[idx].mean(1) / bd[idx].mean(1))
    win = int(((aa == 1) & (bb == 0)).sum())
    lose = int(((aa == 0) & (bb == 1)).sum())
    ci = np.quantile(delta, [.025, .975]).tolist()
    return {"accuracy_difference": float((aa - bb).mean()), "accuracy_difference_ci95": ci,
            "dense_2N_ratio": float(ad.mean() / bd.mean()), "dense_2N_ratio_ci95": np.quantile(dense, [.025, .975]).tolist(),
            "pgm_only_correct": win, "baseline_only_correct": lose,
            "mcnemar_exact_p": float(binomtest(win, win + lose, .5).pvalue) if win + lose else 1.,
            "bootstrap_replicates": 10000, "unit": "question",
            "practical_target_pass": bool((aa - bb).mean() >= -.01 - 1e-12 and ac.mean() / bc.mean() <= .9),
            "statistical_noninferiority_1pp": bool(ci[0] >= -.01 - 1e-12),
            "output_token_ratio": float(ac.mean() / bc.mean()), "output_token_ratio_ci95": np.quantile(ratio, [.025, .975]).tolist()}


def holm(pvalues):
    names = sorted(pvalues, key=pvalues.get)
    n = len(names)
    result, previous = {}, 0.
    for rank, name in enumerate(names):
        value = min(1., max(previous, pvalues[name] * (n - rank)))
        result[name] = value
        previous = value
    return result
