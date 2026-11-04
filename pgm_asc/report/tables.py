"""results/README.md (all tables) and results/summary.csv (one row per cell and method) from results/evaluations/."""
from __future__ import annotations

import csv

from pgm_asc.baselines import METHODS
from pgm_asc.config import MODELS, RESULTS, STUDY, cells
from pgm_asc.model import ARMS
from pgm_asc.report.figures import load_evaluations

TASK_LABEL = {"math": "MATH500", "gsm8k": "GSM8K", "omnimath": "Omni-MATH", "math_to_omnimath": "MATH → Omni-MATH"}
MAIN = [("sc16", "SC16"), ("reasc_offline", "ReASC offline"), ("asc_dev_tuned", "ASC dev-tuned"),
        ("mixture_s42__selected", "PGM mixture"), ("mixture_s42__fixed95", "PGM mixture, γ=0.95")]
HEADS = list(ARMS) + ["same_head_single"]


def cell_value(s):
    return f"{100 * s['correct']:.2f}% / {s['mean_output_tokens']:,.0f}"


def main_table(evaluations, tasks):
    lines = ["| Model | Evaluation | N | " + " | ".join(label for _, label in MAIN) + " |",
             "|---|---|---:|" + "---:|" * len(MAIN)]
    for cell in cells():
        if cell.task in tasks and cell.name in evaluations:
            e = evaluations[cell.name]
            s = e["summary"]
            lines.append(f"| {MODELS[cell.model]['name']} | {TASK_LABEL[cell.task]} | {e['questions']} | "
                         + " | ".join(cell_value(s[k]) for k, _ in MAIN) + " |")
    return lines


def detail(cell, e):
    s = e["summary"]
    lines = [f"### {MODELS[cell.model]['name']}, {TASK_LABEL[cell.task]}", "",
             "| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        v = s[m]
        lines.append(f"| {m} | {100 * v['correct']:.2f}% | {v['samples']:.2f} | {v['mean_output_tokens']:,.0f} | | | |")
    for arm in HEADS:
        a, b = s[f"{arm}_s42__selected"], s[f"{arm}_s42__fixed95"]
        lines.append(f"| {arm} | {100 * a['correct']:.2f}% | {a['samples']:.2f} | {a['mean_output_tokens']:,.0f} | "
                     f"{100 * b['correct']:.2f}% | {b['samples']:.2f} | {b['mean_output_tokens']:,.0f} |")
    lines += ["", "Mixture (selected) minus control (selected), paired by question:", "",
              "| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |", "|---|---:|---|---:|---:|"]
    for name, c in e["primary_structure_contrasts"].items():
        lo, hi = c["accuracy_difference_ci95"]
        lines.append(f"| {name} | {100 * c['accuracy_difference']:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] | "
                     f"{c['output_token_ratio']:.3f} | {c['holm_p_four_primary_structure_contrasts']:.4f} |")
    return lines + [""]


def export():
    evaluations = load_evaluations()
    significant = [f"{name}: mixture vs {k}" for name, e in evaluations.items()
                   for k, c in e["primary_structure_contrasts"].items() if c["holm_p_four_primary_structure_contrasts"] < .05]
    n_contrasts = sum(len(e["primary_structure_contrasts"]) for e in evaluations.values() if not e["transfer"])
    lines = ["# Results", "",
             f"Primary training seed {STUDY['primary_seed']} (seeds 43/44 are in the JSON files). Each entry is **accuracy / mean "
             "output tokens per question**; output tokens = generated tokens + both answer-extraction outputs (input processing "
             "and the teacher-forcing pass are excluded). PGM mixture and ASC dev-tuned use thresholds chosen on dev "
             "(fewest tokens with dev accuracy ≥ dev SC16 − 1 pp).", "",
             "## In-domain", "", "Policies are fitted on the same dataset (calibration/train/dev = 128/800/200 questions).", "",
             *main_table(evaluations, ("math", "gsm8k", "omnimath")), "",
             "## Transfer: MATH → Omni-MATH", "",
             "Each model's MATH policy (heads, calibrators, thresholds, ReASC offline calibration) is replayed unchanged on the "
             "Omni-MATH test bank, without Omni labels.", "",
             *main_table(evaluations, ("math_to_omnimath",)), "",
             "## Structure contrasts", "",
             f"{len([s for s in significant if 'to_omnimath' not in s])} of {n_contrasts} in-domain primary contrasts favor mixture at "
             f"per-cell Holm p < 0.05" + (f" ({'; '.join(significant)})." if significant else "."),
             "Different dev-selected thresholds change cost, so accuracy contrasts are not equal-budget comparisons.", "",
             "## All methods and ablations", ""]
    for cell in cells():
        if cell.name in evaluations:
            lines += detail(cell, evaluations[cell.name])
    lines += ["Figures: [figures/](figures/). Full metrics, seed stability, fixed-prefix metrics and threshold curves: "
              "[evaluations/](evaluations/). Per-question decisions are in the Hugging Face dataset (`decisions/`)."]
    (RESULTS / "README.md").write_text("\n".join(lines) + "\n")
    with (RESULTS / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["cell", "model", "task", "method", "seed", "operating_point", "accuracy", "mean_responses", "mean_output_tokens"])
        for cell in cells():
            for key, v in evaluations.get(cell.name, {}).get("summary", {}).items():
                head, _, op = key.partition("__")
                method, seed = (head.rsplit("_s", 1) + [""])[:2] if op else (head, "")
                writer.writerow([cell.name, cell.model, cell.task, method, seed, op or "default",
                                 f"{v['correct']:.6f}", f"{v['samples']:.4f}", f"{v['mean_output_tokens']:.3f}"])
