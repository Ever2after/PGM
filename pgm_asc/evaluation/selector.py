"""Locked selector holdout evaluation, diagnostics, and plots."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from pgm_asc.core.integrity import verify_lock
from pgm_asc.core.io import log, read_json, read_jsonl, utc_now, write_json, write_jsonl
from pgm_asc.evaluation.metrics import bootstrap_difference, measurements
from pgm_asc.evaluation.replay import sc_accuracy
from pgm_asc.experiments.selector import evaluate_candidate
from pgm_asc.experiments.selector_context import Context
from pgm_asc.policies.calibration import calibrate


def evaluation_report(root, result):
    final = result["final_stage"]
    lines = ["# Selector-stage results", "",
             f"{result['questions']} new questions; all stage policies locked before holdout collection/evaluation.", "",
             "Generated-token replay results. End-to-end online latency for v2 has not been measured.", "",
             "## PGM additions", "",
             "![Accuracy and generated tokens](./accuracy_tokens.png)", "",
             "| Stage | Adopted on calibration | Holdout accuracy | Mean responses | Mean tokens | Min / patience / cap |", "|---|---|---:|---:|---:|---|"]
    for stage, methods in result["stages"].items():
        row = methods["mixture"]
        p = row['policy']
        lines.append(f"| {stage} | {row['adopted_on_calibration']} | {row['accuracy']:.4f} | {row['mean_samples']:.3f} | {row['mean_tokens']:.1f} | {p['minimum']} / {p['patience']} / {p['cap']} |")
    lines += ["", "## Same-opportunity final controls", "",
              "| Method | Accuracy | Mean responses | Mean tokens | SC32-minus-1pp point constraint |", "|---|---:|---:|---:|---|"]
    for method, row in result["stages"][final].items():
        lines.append(f"| {method} | {row['accuracy']:.4f} | {row['mean_samples']:.3f} | {row['mean_tokens']:.1f} | {row['test_point_constraint_met']} |")
    for method, row in result["fixed_budget"].items():
        lines.append(f"| {method} | {row['accuracy']:.4f} | {row['mean_samples']:.3f} | {row['mean_tokens']:.1f} | — |")
    lines += ['', 'Calibration-B strongest control: ' + result['strongest_adaptive_control'] + '.',
              '', 'ReASC retains S4 at S5/S6; no matched continuation learner was fitted for ReASC.']
    lines += ["", "## Paired final comparisons", "", "```json", __import__("json").dumps(result["paired_final_contrasts"], indent=2), "```", "",
              "No test-driven stage promotion. Point constraints and bootstrap intervals are descriptive validation, not a risk guarantee.",
              "Minimum/patience/cap and continuation improvements shared by controls do not establish a PGM-specific mechanism.",
              "Repeated stage comparisons are exploratory; question-level resampling preserves prefix dependence.",
              "Feature extraction/head overhead is excluded from generated tokens; v2 latency remains unverified."]
    lines += ['', '## Remaining PGM failure modes', '', '```json',
              __import__('json').dumps(result['failure_diagnostics']['mixture'], indent=2), '```']
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def failure_diagnostics(rows, history, data, selected):
    gold = data['gold_groups'].numpy()
    counts = {'wrong': 0, 'wrong_with_gold_observed': 0, 'wrong_before_gold_observed': 0,
              'recoverable_by_same_selector_at_32': 0, 't1_stops': 0, 't1_wrong': 0,
              'reached_cap': 0, 'reached_cap_wrong': 0}
    cap_tokens = 0
    for qi, row in enumerate(rows):
        index = qi * 32 + row['t'] - 1
        wrong = not row['correct']
        counts['wrong'] += int(wrong)
        observed = not bool(gold[index, 32])
        counts['wrong_with_gold_observed'] += int(wrong and observed)
        counts['wrong_before_gold_observed'] += int(wrong and not observed)
        last = int(history[qi]['chosen'][-1])
        final_ok = last >= 0 and bool(gold[qi * 32 + 31, last])
        counts['recoverable_by_same_selector_at_32'] += int(wrong and row['t'] < 32 and final_ok)
        counts['t1_stops'] += int(row['t'] == 1)
        counts['t1_wrong'] += int(row['t'] == 1 and wrong)
        at_cap = row['t'] == selected['policy']['cap']
        counts['reached_cap'] += int(at_cap)
        counts['reached_cap_wrong'] += int(at_cap and wrong)
        cap_tokens += row['token_cost'] if at_cap else 0
    counts['fraction_of_tokens_at_cap'] = cap_tokens / max(sum(r['token_cost'] for r in rows), 1)
    return counts


def plot_results(root, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pgm = [m['mixture'] for m in result['stages'].values()]
    ax.plot([r['mean_tokens'] / 1000 for r in pgm], [100 * r['accuracy'] for r in pgm],
            'o-', label='PGM stages', linewidth=1.5, color='tab:blue')
    for i, row in enumerate(pgm):
        ax.annotate(f'S{i}', (row['mean_tokens'] / 1000, 100 * row['accuracy']),
                    xytext=(4, 7 + (i % 3) * 8), textcoords='offset points', fontsize=8)
    for i, (method, row) in enumerate(result['stages'][result['final_stage']].items()):
        if method == 'mixture':
            continue
        ax.scatter(row['mean_tokens'] / 1000, 100 * row['accuracy'], marker='x', s=45)
        ax.annotate(method, (row['mean_tokens'] / 1000, 100 * row['accuracy']),
                    xytext=(5, -12 - (i % 2) * 9), textcoords='offset points', fontsize=8)
    fixed = list(result['fixed_budget'].values())
    ax.plot([r['mean_tokens'] / 1000 for r in fixed], [r['accuracy'] * 100 for r in fixed],
            's--', label='Fixed-budget SC', color='black', alpha=.6)
    ax.set(xlabel='Mean generated tokens (thousands)', ylabel='Accuracy (%)',
           title=f"Locked v2 holdout: {result['questions']} questions")
    ax.grid(alpha=.2)
    ax.legend(loc='best')
    fig.text(.5, .01, 'Point estimates; feature extraction cost and online latency excluded.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(root / 'accuracy_tokens.png', dpi=180)
    fig.savefig(root / 'accuracy_tokens.pdf')
    plt.close(fig)


def evaluate_holdout(config, context=None, force=False):
    root = Path(config["run_dir"])
    lock = read_json(root / "selection_lock.json")
    verify_lock(lock)
    if (root / "evaluation.json").exists() and not force:
        result = read_json(root / "evaluation.json")
        if result["protocol_hash"] != lock["protocol_hash"]:
            raise ValueError("Holdout was already evaluated with a different lock")
        if not (root / 'accuracy_tokens.png').exists() or not (root / 'accuracy_tokens.pdf').exists():
            plot_results(root, result)
        if not (root / 'report.md').exists():
            evaluation_report(root, result)
        return result
    ctx = context if context is not None else Context(config)
    ctx.reasc_settings = lock["reasc_settings"]
    data, metadata = ctx.data("test")
    questions = read_jsonl(Path(config["data_dir"]) / "test.jsonl")
    gold = data["gold_groups"].numpy()
    sc = sc_accuracy(data, metadata)
    nquestions = len(questions)
    result = {"created_at": utc_now(), "protocol_hash": lock["protocol_hash"], "questions": nquestions,
              "stages": {}, "fixed_budget": {}, "paired_stage_contrasts": {}, "paired_final_contrasts": {},
              "final_stage": config["followup_stages"][-1], "online_latency_verified": False,
              "strongest_adaptive_control": lock["strongest_adaptive_control"]}
    all_rows = {}
    for stage, methods in lock["stages"].items():
        result["stages"][stage] = {}
        for method, selection in methods.items():
            before = time.perf_counter()
            metrics = evaluate_candidate(ctx, method, "test", selection["selected"])
            rows = metrics.pop("rows")
            all_rows[(stage, method)] = rows
            write_jsonl(root / "decisions" / f"{stage}_{method}.jsonl", rows)
            result["stages"][stage][method] = {**metrics, "adopted_on_calibration": selection["adopted"],
                                                'policy': selection['selected']['policy'],
                                                'model_variant': selection['selected']['variant'],
                                                'calibrator_kind': (selection['selected']['calibrator'] or {}).get('kind'),
                                                "test_point_constraint_met": metrics["accuracy"] >= sc - .01 - 1e-12,
                                                "prediction_and_replay_seconds": time.perf_counter() - before}
            if stage != config["followup_stages"][0]:
                previous_stage = config["followup_stages"][config["followup_stages"].index(stage) - 1]
                previous = all_rows[(previous_stage, method)]
                result["paired_stage_contrasts"][f"{stage}_{method}"] = {
                    field: bootstrap_difference([r[field] for r in rows], [r[field] for r in previous])
                    for field in ["correct", "token_cost"]}
        log("v2 locked holdout stage evaluated", stage=stage)
    sc_rows = {}
    for t in [8, 16, 32]:
        rows = []
        for qi in range(nquestions):
            index = qi * 32 + t - 1
            leader = metadata[index]["leader"]
            valid = bool(data["candidate_mask"][index, :32].any())
            rows.append({"correct": valid and bool(gold[index, leader]), "token_cost": metadata[index]["token_cost"]})
        sc_rows[t] = rows
        result["fixed_budget"][f"sc{t}"] = {"accuracy": float(np.mean([r["correct"] for r in rows])),
                                                    "mean_tokens": float(np.mean([r["token_cost"] for r in rows])),
                                                    "mean_samples": float(t)}
    final = result["final_stage"]
    pgm = all_rows[(final, "mixture")]
    for method in ["sc32", lock["strongest_adaptive_control"], "single", "random", "alpha1"]:
        control = sc_rows[32] if method == "sc32" else all_rows[(final, method)]
        result["paired_final_contrasts"][f"mixture_minus_{method}"] = {
            field: bootstrap_difference([r[field] for r in pgm], [r[field] for r in control])
            for field in ["correct", "token_cost"]}
    result['failure_diagnostics'] = {}
    result['fixed_budget_probability_metrics'] = {}
    final_nll = {}
    for method in config['followup_methods']:
        selected = lock['stages'][final][method]['selected']
        history = ctx.trace(method, 'test', selected)
        result['failure_diagnostics'][method] = failure_diagnostics(all_rows[(final, method)], history, data, selected)
        if method not in ['asc', 'reasc']:
            raw = ctx.raw(method, 'test', selected['variant'])
            lp = calibrate(raw, data['t'].numpy(), selected['calibrator'])
            result['fixed_budget_probability_metrics'][method] = {
                'raw': measurements(raw, data, 32)[0], 'calibrated': measurements(lp, data, 32)[0]}
            nll = -lp[np.arange(len(lp)), data['target'].numpy()]
            final_nll[method] = nll.reshape(nquestions, 32)[:, [7, 15, 31]].mean(1)
    result['final_calibrated_nll_contrasts'] = {
        method: bootstrap_difference(final_nll['mixture'], final_nll[method])
        for method in ['single', 'random', 'alpha1', 'candidate_mlp', 'cges']}
    subgroup = {}
    for flag in [False, True]:
        mask = np.asarray([q["contains_asy"] == flag for q in questions])
        subgroup[str(flag)] = {"questions": int(mask.sum()), "methods": {}}
        for method in config["followup_methods"]:
            rows = all_rows[(final, method)]
            subgroup[str(flag)]["methods"][method] = {
                "accuracy": float(np.mean([r["correct"] for r, use in zip(rows, mask) if use])) if mask.any() else None,
                "mean_tokens": float(np.mean([r["token_cost"] for r, use in zip(rows, mask) if use])) if mask.any() else None}
    result["asy_subgroups"] = subgroup
    result["bank_audit"] = read_json(root / "test/bank_audit.json")
    verify_lock(lock)
    write_json(root / "evaluation.json", result)
    plot_results(root, result)
    evaluation_report(root, result)
    return result
