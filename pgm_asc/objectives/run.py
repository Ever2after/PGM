"""Fit on common train questions; lock dev choices before any test scoring."""
from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from pgm_asc import config
from pgm_asc.diagnostics import metrics
from pgm_asc.diagnostics.predictive import forecast
from pgm_asc.objectives import CONFIG
from pgm_asc.objectives.train import Counts, Objective, SPEC, common_training_subset, fit_one, prefix_weights
from pgm_asc.utils import digest, file_sha256, utc_now, write_json
from pgm_asc.v2.model import Grid, from_json, log_theta, posterior, readable


def load_splits(model, dataset, splits):
    path = config.ARTIFACTS / "v2" / f"{model}_{dataset}.npz"
    with np.load(path, allow_pickle=False) as z:
        # No response features or gold outside the requested splits are read.
        data = {s: {k: z[f"{s}.{k}"] for k in ("ids", "valid", "correct", "question_id")} for s in splits}
    return data, file_sha256(path)


def objectives(study):
    yield Objective("joint")
    for weighting in study["training"]["weightings"]:
        yield Objective("prefix", weighting)
        for mixing in study["training"]["lambda_grid"]:
            yield Objective("mixed", weighting, mixing)


def identity(study):
    return digest({"study": study, "training_code": file_sha256(Path(__file__).with_name("train.py")),
                   "run_code": file_sha256(Path(__file__)),
                   "model_code": file_sha256(config.ROOT / "pgm_asc" / "v2" / "model.py")})[:16]


def dev_score(counts, params, prefixes):
    with torch.no_grad():
        _, nll = counts.state_losses(from_json(params), Grid())
        return float(nll[:, np.array(prefixes)-1].mean())


def fit_cell(model, dataset, study, out, overwrite=False):
    """This function never loads the test split or its scores."""
    data, source_hash = load_splits(model, dataset, ("train", "dev"))
    key = identity(study)
    path = config.ARTIFACTS / "objective_ablation" / key / "fits" / f"{model}_{dataset}.json"
    if path.exists() and not overwrite:
        cached = json.loads(path.read_text())
        if cached["bank_sha256"] != source_hash or cached["experiment_id"] != key:
            raise ValueError("Fit cache provenance differs; use a new output or --overwrite")
        write_json(Path(out) / "selection" / f"{model}_{dataset}.json", cached)
        return cached
    train, keep = common_training_subset(data["train"])
    dev, dev_keep = common_training_subset(data["dev"])
    if not study["training"]["unique_correct_only"]:
        raise ValueError("This controlled ablation requires the common unique-correct train subset")
    counts, dev_counts = Counts(train), Counts(dev)
    fit_results = {}
    for objective in objectives(study):
        restarts = []
        for seed in study["training"]["restart_seeds"]:
            r = fit_one(counts, objective, study["training"], seed)
            r["dev_unique_nll"] = dev_score(dev_counts, r["params"], study["evaluation"]["prefixes"])
            restarts.append(r)
        best = min(restarts, key=lambda r: (r["penalized_loss"], r["restart_seed"]))
        fit_results[objective.name] = {"objective": vars(objective), "best": best, "restarts": restarts}
        print(f"fit {model}/{dataset}/{objective.name}: dev NLL={best['dev_unique_nll']:.5f}", flush=True)
    selected = {"joint": "joint"}
    for weighting in study["training"]["weightings"]:
        selected[f"prefix_{weighting}"] = f"prefix_{weighting}"
        candidates = [obj.name for obj in objectives(study) if obj.kind == "mixed" and obj.weighting == weighting]
        selected[f"mixed_{weighting}"] = min(candidates, key=lambda name: (
            fit_results[name]["best"]["dev_unique_nll"], fit_results[name]["objective"]["mixing"]))
    result = {"experiment_id": key, "model": model, "dataset": dataset, "bank_sha256": source_hash,
              "study": study, "train_questions": len(keep), "train_kept": int(keep.sum()),
              "dev_questions": len(dev_keep), "dev_unique": int(dev_keep.sum()),
              "train_question_hash": digest(train["question_id"].tolist()),
              "train_question_ids": train["question_id"].tolist(),
              "selection": "restart: penalized train objective; mixed lambda: dev unique NLL at fixed evaluation prefixes",
              "selected": selected, "fits": fit_results, "selection_locked_at": utc_now()}
    write_json(path, result)
    # Reviewable lock is written before evaluate_cell is allowed to load test.
    write_json(Path(out) / "selection" / f"{model}_{dataset}.json", result)
    return result


def average_forecasts(draws):
    result = {}
    for name in draws[0]:
        stacked = np.stack([d[name] for d in draws])
        result[name] = np.sqrt(np.sum(stacked**2, axis=0)) / len(draws) if name.endswith("_mc_se") else np.mean(stacked, axis=0)
    return result


def future_calibration(future, arrays, prefixes, horizons):
    chosen = metrics.reference(arrays)
    hit = metrics.labels(arrays, chosen)
    rows = []
    for pi, t in enumerate(prefixes):
        for hi, h in enumerate(horizons):
            if t+h > 16:
                continue
            truth = {"repair": ~hit[:, t-1] & hit[:, t+h-1], "damage": hit[:, t-1] & ~hit[:, t+h-1]}
            for event in truth:
                for b in metrics.reliability(future[event][:, pi, hi], truth[event]):
                    rows.append({"prefix": t, "horizon": h, "event": event, **b})
    return rows


def evaluate_cell(model, dataset, study, out, fitted, overwrite=False):
    out, key = Path(out), identity(study)
    path = out / "cells" / f"{model}_{dataset}.json"
    if path.exists() and not overwrite:
        old = json.loads(path.read_text())
        if old["experiment_id"] == key and old["fit_digest"] == digest(fitted):
            return str(path)
        raise ValueError("Existing evaluation belongs to different fits/configuration")
    data, bank_hash = load_splits(model, dataset, study["evaluation"]["splits"])
    if bank_hash != fitted["bank_sha256"]:
        raise ValueError("Bank changed after fitting")
    e = study["evaluation"]
    grid = Grid()
    params_by_name = {name: fitted["fits"][source]["best"]["params"] for name, source in fitted["selected"].items()}
    original_path = config.RESULTS / "v2" / "cells" / f"{model}_{dataset}.json"
    original = json.loads(original_path.read_text())
    params_by_name["released_M1"] = original["fitted"]["M0"]["params"]
    result = {"experiment_id": key, "model": model, "dataset": dataset, "study": study,
              "bank_sha256": bank_hash, "released_result_sha256": file_sha256(original_path),
              "fit_digest": digest(fitted), "selection": fitted["selected"],
              "train_kept": fitted["train_kept"], "selection_locked_at": fitted["selection_locked_at"],
              "evaluation_started_at": utc_now(), "splits": {},
              "parameters": {name: {"raw": p, "readable": readable(p), "count": 4} for name, p in params_by_name.items()}}
    cache_values = {}
    for split, arrays in data.items():
        entries, per_model = {}, {}
        unique = metrics.targets(arrays)[3]
        for name, raw_params in params_by_name.items():
            params = from_json(raw_params)
            observations = {k: v for k, v in arrays.items() if k != "correct"}
            p = posterior(params, SPEC, observations, {}, keep_joint=True)
            theta = np.exp(log_theta(params, SPEC, grid).detach().numpy())
            seed_draws, seed_summaries = [], []
            for seed in e["mc_seeds"]:
                f = forecast(observations, p, grid.q.numpy(), theta, e["prefixes"], e["horizons"],
                             e["simulations_per_seed"], seed, f"{model}/{dataset}/{split}",
                             paired_gain=True, transition_details=True)
                _, _, fl = metrics.future_scores(f, arrays, e["prefixes"], e["horizons"])
                seed_summaries.append({"seed": seed, **{k: float(v.mean()) for k, v in fl.items()}})
                seed_draws.append(f)
            future = average_forecasts(seed_draws)
            assert np.allclose(future["gain"], future["repair"]-future["damage"], equal_nan=True)
            pr, pb, pl = metrics.posterior_losses(p.post, arrays, e["prefixes"])
            fr, gb, fl = metrics.future_scores(future, arrays, e["prefixes"], e["horizons"])
            losses = {**pl, **fl}
            summary = {k: metrics.bootstrap_mean(v, e["bootstrap_replicates"], e["bootstrap_seed"]) for k, v in losses.items()}
            unique_summary = {k: metrics.bootstrap_mean(v[unique], e["bootstrap_replicates"], e["bootstrap_seed"]) for k, v in fl.items()}
            entries[name] = {"summary": summary, "future_unique_summary": unique_summary,
                             "posterior_by_prefix": pr, "posterior_calibration": pb,
                             "future_by_horizon": fr, "gain_calibration": gb,
                             "transition_calibration": future_calibration(future, arrays, e["prefixes"], e["horizons"]),
                             "mc_seed_sensitivity": seed_summaries}
            per_model[name] = losses
            cache_values[f"{split}.{name}.post"] = p.post
            for k, v in future.items():
                cache_values[f"{split}.{name}.future.{k}"] = v
            print(f"evaluate {model}/{dataset}/{split}/{name}: paired MC complete", flush=True)
        comparisons = {}
        for name, losses in per_model.items():
            if name == "joint":
                continue
            comparisons[name] = {k: metrics.bootstrap_mean(value-per_model["joint"][k], e["bootstrap_replicates"], e["bootstrap_seed"])
                                 for k, value in losses.items()}
        result["splits"][split] = {"questions": len(unique), "unique_questions": int(unique.sum()),
                                  "models": entries, "paired_model_minus_joint": comparisons}
        cache_values[f"{split}.question_id"] = arrays["question_id"]
    cache = config.ARTIFACTS / "objective_ablation" / key / "predictions" / f"{model}_{dataset}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    temp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **cache_values)
    temp.replace(cache)
    result["prediction_cache"] = {"path": str(cache), "sha256": file_sha256(cache)}
    write_json(path, result)
    return str(path)


def _job(job):
    model, dataset, study, out, stage, overwrite = job
    fitted = fit_cell(model, dataset, study, out, overwrite)
    if stage == "fit":
        return model, dataset
    return evaluate_cell(model, dataset, study, out, fitted, overwrite)


def run(models=None, datasets=None, workers=2, out=None, stage="run", overwrite=False, study=None):
    study = copy.deepcopy(study or CONFIG)
    models, datasets = models or study["models"], datasets or study["datasets"]
    out = Path(out or config.RESULTS / "objective_ablation").resolve()
    if stage == "report":
        from pgm_asc.objectives.report import report
        return report(out)
    protocol = {"experiment_id": identity(study), "study": study,
                "model": {"parameters": 4, "inputs": ["answer ids", "valid"], "graph": "M1 unchanged"},
                "prefix_weights": {w: prefix_weights(w).tolist() for w in study["training"]["weightings"]},
                "code_sha256": {p.name: file_sha256(p) for p in (Path(__file__), Path(__file__).with_name("train.py"),
                                   config.ROOT / "pgm_asc" / "diagnostics" / "predictive.py",
                                   config.ROOT / "pgm_asc" / "diagnostics" / "metrics.py")}}
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and not overwrite:
        old = json.loads(protocol_path.read_text())
        if old != protocol:
            raise ValueError("Protocol/code differs from existing output; choose --out or --overwrite")
    write_json(protocol_path, protocol)
    jobs = [(m, d, study, str(out), stage, overwrite) for m in models for d in datasets]
    if workers == 1:
        paths = [_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            paths = list(pool.map(_job, jobs))
    if stage != "fit":
        write_json(out / "run.json", {"cells": [str(Path(p).relative_to(out)) for p in paths],
                                     "models": models, "datasets": datasets, "experiment_id": identity(study)})
        from pgm_asc.objectives.report import report
        report(out)
    return paths
