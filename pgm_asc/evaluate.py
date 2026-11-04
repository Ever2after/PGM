"""Replay a saved policy and every baseline on a cell's test bank.

Writes results/evaluations/<cell>.json (metrics, paired contrasts, seed stability, fixed-prefix
metrics, threshold curves) and artifacts/decisions/<cell>.parquet (one row per question and method).
"""
from __future__ import annotations

import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from pgm_asc.baselines import METHODS, baseline_decision, online_calibrate
from pgm_asc.config import EVAL_SPLIT, STUDY, Cell, checkpoint_path, decisions_path, evaluation_path, policy_path
from pgm_asc.data.bank import load_bank
from pgm_asc.data.features import prepare
from pgm_asc.model import predict, restore
from pgm_asc.policies import calibrate, replay_policy, stopping_index, traces
from pgm_asc.stats import decision_row, holm, metrics, paired
from pgm_asc.utils import log, read_json, utc_now, write_json

PRIMARY_CONTROLS = ["alpha1", "random", "candidate_confidence", "deepsets_embedding"]
BASELINE_CONTRASTS = ["sc16", "reasc_offline", "reasc_online", "reasc_dev_tuned"]


def uses_features(head_name):
    """Count-only heads need no teacher-forcing pass at serving time."""
    return not head_name.startswith("candidate_counts_")


def evaluate_cell(cell: Cell) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(4)
    policy = read_json(policy_path(cell.policy_name))
    checkpoints = torch.load(checkpoint_path(cell.policy_name), map_location="cpu", weights_only=True)
    banks = load_bank(cell.model, cell.eval_dataset, EVAL_SPLIT)
    data, meta = prepare(banks, policy["representation"], policy["scalers"])
    n_params = cell.spec["rounded_parameters"]
    offline = policy["reasc_offline"]
    online = online_calibrate([b.records[0]["bottom10_confidence"] for b in banks])
    rows, summary, curves, prefix = {}, {}, {}, {}
    for method in METHODS:
        rows[method] = [decision_row(b, *baseline_decision(b, method, offline, online, policy["dev_tuned_baselines"]), False, n_params)
                        for b in banks]
        summary[method] = metrics(rows[method])
    gold = data["gold_groups"].numpy()
    target = data["target"].numpy()
    for name, head in policy["heads"].items():
        model = restore(checkpoints[name])
        lp = calibrate(predict(model, data), head["calibrator"])
        history = traces(lp, data)
        prefix[name] = {}
        for t in [2, 4, 8, 16]:
            indices = np.arange(t - 1, len(lp), 16)
            chosen = lp[indices, :32].argmax(1)
            prefix[name][str(t)] = {"accuracy": float(gold[indices, chosen].mean()),
                                    "nll_first_correct_or_OTHER": float(-lp[indices, target[indices]].mean())}
        for op in ["selected", "fixed95"]:
            stop = head["selected"]["policy"] if op == "selected" else head["fixed95"]
            rr = []
            for b, h in zip(banks, history):
                i, why = stopping_index(h, stop)
                rr.append(decision_row(b, i, int(h["chosen"][i]), why, uses_features(name), n_params))
            rows[f"{name}__{op}"] = rr
            summary[f"{name}__{op}"] = metrics(rr)
        curves[name] = []
        for gamma in STUDY["gammas"]:
            r = replay_policy(history, data, meta, {**STUDY["stopping"], "gamma": gamma})
            curves[name].append({"gamma": gamma, **{k: v for k, v in r.items() if k != "rows"}})
    seed = STUDY["primary_seed"]
    mixture = rows[f"mixture_s{seed}__selected"]
    contrasts = {c: paired(mixture, rows[f"{c}_s{seed}__selected"]) for c in PRIMARY_CONTROLS}
    adjusted = holm({k: v["mcnemar_exact_p"] for k, v in contrasts.items()})
    for k, v in contrasts.items():
        v["holm_p_four_primary_structure_contrasts"] = adjusted[k]
    stability = {}
    for arm in sorted({name.rsplit("_s", 1)[0] for name in policy["heads"]}):
        for op in ["selected", "fixed95"]:
            values = [summary[f"{arm}_s{s}__{op}"] for s in STUDY["train_seeds"]]
            stability[f"{arm}__{op}"] = {k: {"mean": float(np.mean([v[k] for v in values])),
                                            "std_across_3_seeds": float(np.std([v[k] for v in values], ddof=1))}
                                        for k in ["correct", "mean_output_tokens", "samples"]}
    result = {
        "cell": cell.name, "model": cell.spec["model_id"], "fit_dataset": cell.fit_dataset, "eval_dataset": cell.eval_dataset,
        "policy": cell.policy_name, "transfer": cell.transfer, "questions": len(banks), "created_at": utc_now(),
        "summary": summary, "primary_structure_contrasts": contrasts,
        "baseline_contrasts": {k: paired(mixture, rows[k]) for k in BASELINE_CONTRASTS},
        "seed_stability": stability, "fixed_prefix_metrics": prefix, "threshold_curves": curves,
        "reasc_online_calibration": online, "seconds": time.perf_counter() - started,
        "notes": {"cost": "Output tokens = generated tokens + both answer-extraction outputs. Input processing and the "
                          "teacher-forcing feature pass are excluded; bank replay is not serving latency.",
                  "multiple_testing": "Holm adjustment covers the four primary structure contrasts of this cell only."}}
    write_json(evaluation_path(cell.name), result)
    flat = [{"method": method, **r} for method, rr in rows.items() for r in rr]
    path = decisions_path(cell.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(flat), path, compression="zstd")
    log("evaluated", cell=cell.name, seconds=result["seconds"])
    return result
