"""Fit every head of a cell, calibrate it, select its stopping threshold on dev, and save the policy.

Output (artifacts/policies/):
  <model>_<dataset>.json             representation, scalers, calibrators, thresholds, baseline calibration
  <model>_<dataset>.checkpoints.pt   state dicts of all 36 heads (11 arms x 3 seeds + same-head control)
"""
from __future__ import annotations

import time

import torch

from pgm_asc.baselines import offline_calibrate, sc_accuracy, tune_thresholds
from pgm_asc.config import STUDY, Cell, checkpoint_path, policy_path
from pgm_asc.data.bank import load_bank
from pgm_asc.data.features import attach, build, fit_representation, fit_scalers, scale
from pgm_asc.model import ARMS, SAME_HEAD_SINGLE, predict, restore
from pgm_asc.policies import calibrate, global_calibrator, replay_policy, select_candidate, traces
from pgm_asc.train import fit
from pgm_asc.utils import log, utc_now, write_json


def threshold_candidates(model, calibrator, data, meta):
    histories = traces(calibrate(predict(model, data), calibrator), data)
    rows = []
    for gamma in STUDY["gammas"]:
        policy = {**STUDY["stopping"], "gamma": gamma}
        result = replay_policy(histories, data, meta, policy)
        rows.append({k: v for k, v in result.items() if k != "rows"} | {"policy": policy})
    return rows


def fit_cell(cell: Cell, overwrite: bool = False) -> dict:
    if cell.transfer:
        raise SystemExit(f"{cell.name} replays the {cell.policy_name} policy; fit that cell instead.")
    out, ckpt = policy_path(cell.policy_name), checkpoint_path(cell.policy_name)
    if out.exists() and not overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to refit.")
    started = time.perf_counter()
    torch.set_num_threads(4)
    banks = {s: load_bank(cell.model, cell.fit_dataset, s) for s in ("train", "dev", "calibration")}
    representation = fit_representation(banks["train"])
    data, meta, scalers = {}, {}, None
    for split in ("train", "dev", "calibration"):
        raw, m = build(banks[split], representation)
        if split == "train":
            scalers = fit_scalers(raw)
        data[split], meta[split] = attach(scale(raw, scalers), banks[split], m, representation)
    sc = sc_accuracy(banks["dev"])
    tolerance = STUDY["dev_accuracy_tolerance"]
    offline = offline_calibrate([b.records[0]["lowest_confidence"] for b in banks["calibration"]],
                                [b.labels["raw_response_correct"][0] for b in banks["calibration"]])
    fixed = {**STUDY["stopping"], "gamma": STUDY["fixed_gamma"]}
    heads, checkpoints = {}, {}

    def add(name, arm, seed, model, entry, summary):
        calibrator = global_calibrator(predict(model, data["calibration"]), data["calibration"])
        candidates = threshold_candidates(model, calibrator, data["dev"], meta["dev"])
        heads[name] = {"arm": arm, "seed": seed, "spec": entry["spec"], "calibrator": calibrator,
                       "selected": select_candidate(candidates, sc, tolerance), "fixed95": fixed,
                       "candidates": candidates, "fit": summary}
        checkpoints[name] = entry

    for seed in STUDY["train_seeds"]:
        for arm, spec in ARMS.items():
            model, entry, summary, _ = fit(data["train"], data["dev"], spec, STUDY["train"], seed, label=f"{cell.name}/{arm}")
            add(f"{arm}_s{seed}", arm, seed, model, entry, summary)
        # Same-head control: the trained mixture restricted to partition 1, without retraining.
        mixture = checkpoints[f"mixture_s{seed}"]
        entry = {**mixture, "spec": SAME_HEAD_SINGLE}
        add(f"same_head_single_s{seed}", "same_head_single", seed, restore(entry), entry, {"from": f"mixture_s{seed}"})
        log("seed complete", cell=cell.name, seed=seed, heads=len(heads))
    policy = {"policy": cell.policy_name, "model": cell.model, "model_id": cell.spec["model_id"], "dataset": cell.fit_dataset,
              "created_at": utc_now(), "study": {k: STUDY[k] for k in ("train", "train_seeds", "primary_seed", "stopping", "gammas", "fixed_gamma", "dev_accuracy_tolerance")},
              "representation": representation, "scalers": scalers, "dev_sc16_accuracy": sc, "reasc_offline": offline,
              "dev_tuned_baselines": tune_thresholds(banks["dev"], offline, sc, STUDY["gammas"], tolerance),
              "heads": heads, "fit_seconds": time.perf_counter() - started}
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoints, ckpt)
    write_json(out, policy)
    log("policy saved", cell=cell.name, path=str(out), seconds=policy["fit_seconds"])
    return policy
