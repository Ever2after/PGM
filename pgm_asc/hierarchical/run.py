"""P1 structure study: common train joint objective; numerical checks on dev."""
from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from pgm_asc import config
from pgm_asc.diagnostics import metrics
from pgm_asc.hierarchical import CONFIG
from pgm_asc.hierarchical.model import KEYS, Statistics, forecast, infer, readable, tensors
from pgm_asc.hierarchical.train import fit_one, integration_check
from pgm_asc.objectives.run import average_forecasts, future_calibration, load_splits
from pgm_asc.objectives.train import common_training_subset
from pgm_asc.utils import digest, file_sha256, utc_now, write_json


def code_hashes():
    paths = [Path(__file__).with_name(p) for p in ("model.py", "train.py", "run.py")]
    paths += [config.ROOT / "pgm_asc" / "diagnostics" / p for p in ("metrics.py", "predictive.py")]
    paths += [config.ROOT / "pgm_asc" / "v2" / "model.py",
              config.ROOT / "pgm_asc" / "objectives" / "run.py",
              config.ROOT / "pgm_asc" / "objectives" / "train.py"]
    return {str(p.relative_to(config.ROOT)): file_sha256(p) for p in paths}


def identity(study):
    return digest({"study": study, "code": code_hashes()})[:16]


def fit_cell(model, dataset, study, out, overwrite=False):
    torch.set_num_threads(1)
    data, source_hash = load_splits(model, dataset, ("train", "dev"))
    key = identity(study)
    path = config.ARTIFACTS / "hierarchical_crp" / key / "fits" / f"{model}_{dataset}.json"
    if path.exists() and not overwrite:
        cached = json.loads(path.read_text())
        if cached["bank_sha256"] != source_hash or cached["experiment_id"] != key:
            raise ValueError("Fit cache provenance differs")
        write_json(Path(out) / "selection" / f"{model}_{dataset}.json", cached)
        return cached
    if study["training"]["objective"] != "joint_nll_per_response" or not study["training"]["unique_correct_only"]:
        raise ValueError("P1 comparison requires the common unique-correct joint objective")
    train, keep = common_training_subset(data["train"])
    fits, checks = {}, []
    bound = study["integration"]["sigma_bound"]
    ladder = study["integration"]["normal_nodes"]
    nodes, passed = ladder[0], False
    for ni, nodes in enumerate(ladder[:-1]):
        for variant in study["variants"]:
            if variant in fits and "sigma_raw" not in KEYS[variant]:
                continue
            restarts = [fit_one(train, variant, study["training"], seed, nodes, bound)
                        for seed in study["training"]["restart_seeds"]]
            best = min(restarts, key=lambda r: (r["penalized_loss"], r["restart_seed"]))
            fits[variant] = {"best": best, "restarts": restarts}
            print(f"fit {model}/{dataset}/{variant} u={nodes}: J={best['joint_nll_per_response']:.6f}", flush=True)
        current = []
        for variant in study["variants"]:
            if "sigma_raw" not in KEYS[variant]:
                continue
            check = integration_check(fits[variant]["best"]["params"], variant, train, data["dev"],
                                      nodes, ladder[ni+1], study["evaluation"]["prefixes"], bound)
            current.append({"variant": variant, **check})
        checks.extend(current)
        passed = all(c["max_error"] <= study["integration"]["tolerance"] for c in current)
        print(f"quadrature {model}/{dataset} u={nodes}: max error={max([c['max_error'] for c in current], default=0):.3g}", flush=True)
        if passed:
            break
    if not passed:
        raise RuntimeError(f"Quadrature did not converge for {model}/{dataset}; refine train/dev before test")
    for variant, fit in fits.items():
        p = infer(tensors(fit["best"]["params"]), variant, data["dev"], nodes, bound)
        _, _, loss = metrics.posterior_losses(p.post, data["dev"], study["evaluation"]["prefixes"])
        fit["best"]["dev_unique_nll"] = float(np.nanmean(loss["unique_nll"]))
    result = {"experiment_id": key, "model": model, "dataset": dataset, "bank_sha256": source_hash,
              "study": study, "train_questions": len(keep), "train_kept": int(keep.sum()),
              "train_question_hash": digest(train["question_id"].tolist()), "train_question_ids": train["question_id"].tolist(),
              "selection": "minimum penalized train joint loss; shared numerical resolution checked on train/dev only",
              "normal_nodes": nodes, "quadrature_checks": checks, "fits": fits,
              "selection_locked_at": utc_now()}
    previous = config.RESULTS / "objective_ablation" / "selection" / f"{model}_{dataset}.json"
    if previous.exists() and "M1" in fits:
        old = json.loads(previous.read_text())
        if old["bank_sha256"] != source_hash or old["train_question_hash"] != result["train_question_hash"]:
            raise ValueError("Controlled baseline cohort differs from prior M1 study")
        old_best = old["fits"]["joint"]["best"]
        gap = fits["M1"]["best"]["penalized_loss"]-old_best["penalized_loss"]
        result["previous_M1_joint"] = {"selection_sha256": file_sha256(previous),
                                       "penalized_train_gap": gap,
                                       "max_raw_parameter_gap": max(abs(fits['M1']['best']['params'][k]-v) for k,v in old_best['params'].items())}
        if abs(gap) > 1e-5:
            raise ValueError("M1 baseline no longer reproduces the prior controlled joint fit")
    write_json(path, result)
    write_json(Path(out) / "selection" / f"{model}_{dataset}.json", result)
    return result


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
    torch.set_num_threads(1)
    params_by_name = {name: fit["best"]["params"] for name, fit in fitted["fits"].items()}
    nodes, bound = fitted["normal_nodes"], study["integration"]["sigma_bound"]
    result = {"experiment_id": key, "model": model, "dataset": dataset, "study": study,
              "bank_sha256": bank_hash, "normal_nodes": nodes,
              "fit_digest": digest(fitted), "selection": fitted["selection"],
              "train_kept": fitted["train_kept"], "selection_locked_at": fitted["selection_locked_at"],
              "evaluation_started_at": utc_now(), "splits": {},
              "parameters": {name: {"raw": p, "readable": readable(p, bound), "count": len(p)} for name, p in params_by_name.items()}}
    cache_values = {}
    for split, arrays in data.items():
        entries, per_model = {}, {}
        unique = metrics.targets(arrays)[3]
        for name, raw_params in params_by_name.items():
            params = tensors(raw_params)
            observations = {k: v for k, v in arrays.items() if k != "correct"}
            p = infer(params, name, observations, nodes, bound)
            seed_draws, seed_summaries = [], []
            for seed in e["mc_seeds"]:
                f = forecast(observations, p, e["prefixes"], e["horizons"],
                             e["simulations_per_seed"], seed, f"{model}/{dataset}/{split}")
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
            if name == "M1":
                continue
            comparisons[name] = {k: metrics.bootstrap_mean(value-per_model["M1"][k], e["bootstrap_replicates"], e["bootstrap_seed"])
                                 for k, value in losses.items()}
        result["splits"][split] = {"questions": len(unique), "unique_questions": int(unique.sum()),
                                  "models": entries, "paired_model_minus_M1": comparisons}
        cache_values[f"{split}.question_id"] = arrays["question_id"]
    cache = config.ARTIFACTS / "hierarchical_crp" / key / "predictions" / f"{model}_{dataset}.npz"
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
    return (model, dataset) if stage == "fit" else evaluate_cell(model, dataset, study, out, fitted, overwrite)


def run(models=None, datasets=None, workers=2, out=None, stage="run", overwrite=False, study=None):
    study = copy.deepcopy(study or CONFIG)
    models, datasets = models or study["models"], datasets or study["datasets"]
    out = Path(out or config.RESULTS / "hierarchical_crp").resolve()
    if stage == "report":
        from pgm_asc.hierarchical.report import report
        return report(out)
    if "M1" not in study["variants"] or any(v not in KEYS for v in study["variants"]):
        raise ValueError("Require M1 baseline and known variants")
    if len(study["integration"]["normal_nodes"]) < 2:
        raise ValueError("At least two resolutions required for numerical checks")
    protocol = {"experiment_id": identity(study), "study": study, "code_sha256": code_hashes(),
                "inputs": ["answer ids", "valid"], "parameters": {k: list(KEYS[k]) for k in study["variants"]},
                "graph": "q~Beta(a,b); u~Normal(0,1); logtheta=clip(t0+t1*(2q-1)[+t2*(2q-1)^2],-8,8)+sigma*u",
                "sigma": "abs(3*tanh(sigma_raw/3)); signed loading is equivalent under symmetric Normal quadrature",
                "sampling": "same uniforms per question/prefix/seed; c,q,theta drawn jointly once and held across future responses"}
    path = out / "protocol.json"
    if path.exists() and not overwrite and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol/code differs; use new output or --overwrite")
    write_json(path, protocol)
    jobs = [(m,d,study,str(out),stage,overwrite) for m in models for d in datasets]
    if workers == 1:
        paths = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            paths = list(pool.map(_job, jobs))
    if stage != "fit":
        write_json(out / "run.json", {"cells": [str(Path(p).relative_to(out)) for p in paths],
                                     "models": models, "datasets": datasets, "experiment_id": identity(study)})
        from pgm_asc.hierarchical.report import report
        report(out)
    return paths
