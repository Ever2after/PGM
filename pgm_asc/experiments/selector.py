"""Selector-stage calibration ladder and policy locking."""
from __future__ import annotations

import copy

from pathlib import Path

import joblib

from pgm_asc.core.integrity import file_sha, source_files, verify_lock
from pgm_asc.core.io import digest, log, read_json, utc_now, write_json
from pgm_asc.evaluation.replay import replay_policy, sc_accuracy
from pgm_asc.experiments.selector_context import Context
from pgm_asc.policies.continuation import continuation_predict
from pgm_asc.policies.stopping import GAMMAS, select_candidate
from pgm_asc.training.continuation import fit_continuation


def evaluate_candidate(ctx, method, split, candidate, value_predictions=None):
    data, metadata = ctx.data(split)
    history = ctx.trace(method, split, candidate)
    if candidate.get("continuation_head") and value_predictions is None:
        fitted = joblib.load(candidate["continuation_head"])
        value_predictions = continuation_predict(fitted, history, data)
    result = replay_policy(history, data, metadata, candidate["policy"], value_predictions)
    return result


def policy_grid(ctx, method, template, minimums=None, patiences=None, caps=None, lambdas=None, values=None):
    original = template["policy"]
    candidates = []
    for minimum in minimums or [original["minimum"]]:
        for patience in patiences or [original["patience"]]:
            for cap in caps or [original["cap"]]:
                if minimum > cap:
                    continue
                for cost_lambda in lambdas if lambdas is not None else [None]:
                    for gamma in GAMMAS:
                        candidate = copy.deepcopy(template)
                        candidate["policy"] = {"minimum": minimum, "patience": patience, "cap": cap, "gamma": gamma}
                        if cost_lambda is not None:
                            candidate["policy"]["continuation_lambda"] = cost_lambda
                        metrics = evaluate_candidate(ctx, method, "calibration_b", candidate, values)
                        metrics.pop("rows")
                        candidates.append({**candidate, **metrics})
    return candidates


def selection_report(root, result):
    lines = ["# PGM v2 sequential calibration results", "",
             "Calibration-B only. Holdout has not been used for policy selection.", "",
             "| Stage | Method | Adopted | Accuracy | Mean tokens | Min / patience / cap / gamma |", "|---|---|---|---:|---:|---|"]
    for stage, methods in result["stages"].items():
        for method, row in methods.items():
            selected = row["selected"]
            p = selected["policy"]
            lines.append(f"| {stage} | {method} | {row['adopted']} | {selected['accuracy']:.4f} | {selected['mean_tokens']:.1f} | {p['minimum']} / {p['patience']} / {p['cap']} / {p['gamma']} |")
    lines += ["", "Feasibility is an empirical SC32-minus-1pp calibration constraint; it is not a population guarantee.",
              "Generated-token cost excludes feature extraction and model/head fitting. Latency claims require separate online measurement."]
    (root / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_stages(config, context=None):
    root = Path(config["run_dir"])
    lockpath = root / "selection_lock.json"
    if lockpath.exists():
        locked = read_json(lockpath)
        verify_lock(locked)
        return locked
    ctx = context if context is not None else Context(config)
    data, metadata = ctx.data("calibration_b")
    sc = sc_accuracy(data, metadata)
    result = {"created_at": utc_now(), "cohort_sha256": file_sha(root / "cohort_lock.json"),
              "config_sha256": file_sha("configs/experiment.json"), "calibration_B_sc32": sc,
              "stages": {}, "holdout_accessed": False}
    latest = {}
    for stage in config["followup_stages"]:
        result["stages"][stage] = {}
        for method in config["followup_methods"]:
            log("v2 stage selection", stage=stage, method=method)
            write_json(root / "selection_status.json", {"state": "running", "stage": stage, "method": method,
                                                        "updated_at": utc_now(), "holdout_accessed": False})
            previous = latest.get(method)
            if previous is None:
                template = {"variant": "original", "calibrator": ctx.calibrator(method, "original", "global"),
                            "policy": {"minimum": 1, "patience": 1, "cap": 32, "gamma": None}}
            else:
                template = copy.deepcopy(previous)
                for key in ["accuracy", "mean_tokens", "mean_samples", "feasible"]:
                    template.pop(key, None)
            grid = []
            if stage == "s0_global":
                grid = policy_grid(ctx, method, template)
            elif stage == "s1_minimum":
                grid = policy_grid(ctx, method, template, minimums=config["minimum_grid"])
            elif stage == "s2_patience":
                grid = policy_grid(ctx, method, template, patiences=config["patience_grid"])
            elif stage == "s3_cap":
                grid = policy_grid(ctx, method, template, caps=config["cap_grid"])
            elif stage == "s4_prefix_calibration" and method not in ["asc", "reasc"]:
                template["calibrator"] = ctx.calibrator(method, template["variant"], "bucket")
                grid = policy_grid(ctx, method, template)
            elif stage == "s5_prefix_loss" and method in config["refit_methods"]:
                template["variant"] = "prefix"
                for kind in ["global", "bucket"]:
                    template["calibrator"] = ctx.calibrator(method, "prefix", kind)
                    grid.extend(policy_grid(ctx, method, template))
            elif stage == "s6_continuation" and method != "reasc":
                train, train_meta = ctx.data("train")
                dev, dev_meta = ctx.data("dev")
                headroot = root / "continuation" / method / digest(template)[:16]
                headroot.mkdir(parents=True, exist_ok=True)
                fitted, fit_result = fit_continuation(ctx.trace(method, "train", template), ctx.trace(method, "dev", template),
                                                       train, dev, train_meta, dev_meta, headroot, cap=template['policy']['cap'])
                template["continuation_head"] = str(headroot / "continuation.joblib")
                values = continuation_predict(fitted, ctx.trace(method, "calibration_b", template), data)
                grid = policy_grid(ctx, method, template, lambdas=config["continuation_lambda_grid"], values=values)
            if not grid:
                grid = [copy.deepcopy(previous)]
            selected, adopted = select_candidate(grid, sc, previous)
            latest[method] = copy.deepcopy(selected)
            reason = "initial control" if previous is None else (
                "lower calibration token cost under empirical constraint, or repaired infeasibility" if adopted else
                "kept previous policy; no qualifying cost improvement or stage not applicable")
            result["stages"][stage][method] = {"selected": copy.deepcopy(selected), "adopted": adopted,
                                                "reason": reason, "candidates": grid}
            write_json(root / "selection_progress.json", result)
            selection_report(root, result)
    result["reasc_settings"] = ctx.reasc_settings
    eligible = {m: r for m, r in latest.items() if m != "mixture" and r["feasible"]}
    controls = {m: r for m, r in latest.items() if m != 'mixture'}
    result["any_feasible_control"] = bool(eligible)
    result["strongest_adaptive_control"] = (min(eligible, key=lambda m: eligible[m]["mean_tokens"]) if eligible else
                                             min(controls, key=lambda m: (-controls[m]['accuracy'], controls[m]['mean_tokens'])))
    artifacts = [Path(config.get("config_path", "configs/experiment.json")), Path("docs/METHOD.md"), root / "cohort_lock.json"]
    artifacts += source_files()
    artifacts += list((root / "refits").glob("*/checkpoint.pt"))
    artifacts += list((root / "calibrators").glob("*/*/*.json"))
    artifacts += list((root / "continuation").glob("*/*/continuation.joblib"))
    artifacts += [ctx.source / "initial_selection.json"]
    artifacts += list((ctx.source / "initial").glob("*/checkpoint.pt"))
    artifacts += list((ctx.source / "cache").glob("*"))
    result["artifact_sha256"] = {str(path): file_sha(path) for path in artifacts if path.is_file()}
    verify_lock(result)
    result["locked_at"] = utc_now()
    result["protocol_hash"] = digest(result)
    write_json(lockpath, result)
    write_json(root / "selection_status.json", {"state": "complete", "stage_count": len(result["stages"]),
                                                "updated_at": utc_now(), "holdout_accessed": False})
    log("v2 all stage policies locked", protocol_hash=result["protocol_hash"], strongest=result["strongest_adaptive_control"])
    return result
