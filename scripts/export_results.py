"""Publish aggregate metrics and cohort hashes without banks or model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]
MODELS = ("gemma4b", "qwen3b", "qwen7b")
DISPLAY = {"gemma4b": "Gemma-3-4B-IT", "qwen3b": "Qwen2.5-3B-Instruct", "qwen7b": "Qwen2.5-7B-Instruct"}
BASELINES = ("pass1", "sc16", "asc", "esc", "reasc_offline", "reasc_online", "asc_dev_tuned", "reasc_dev_tuned")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def export(project: Path = PROJECT, output: Path | None = None) -> None:
    project = Path(project)
    target = Path(output) if output is not None else project / "results" / "followup"
    run = project / "artifacts" / "runs" / "pgm_followup"
    cells = {}
    for model in MODELS:
        for dataset, splits in (("math", ("math500", "confirm", "previous_fresh")), ("gsm8k", ("test",))):
            for split in splits:
                path = run / f"{model}_{dataset}" / split / "study_evaluation.json"
                if path.exists():
                    data = json.loads(path.read_text())
                    cells[f"{model}_{dataset}/{split}"] = data
    if not cells:
        raise SystemExit("No completed study_evaluation.json files found; existing exports left untouched")
    target.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for key, data in cells.items():
        source = Path("artifacts/runs/pgm_followup") / key / "study_evaluation.json"
        # Retain all three training seeds, common operating points and statistics.
        # Per-question decisions, generated text and threshold sweeps stay local.
        summaries[key] = {k: data[k] for k in (
            "created_at", "model", "dataset", "split", "questions", "evaluation_role", "summary",
            "primary_structure_contrasts_seed42", "secondary_baseline_contrasts_seed42", "seed_stability",
            "fixed_prefix_metrics", "supervision_budget", "inference_cost_note", "multiple_testing_note")}
        summaries[key].update(source=str(source), source_sha256=digest(project / source))
    write_json(target / "metrics.json", {"schema_version": 1, "primary_seed": 42, "omni_status": "deferred", "evaluations": summaries})
    lines = ["# Follow-up results", "",
             "Aggregate results exported from the completed, locked evaluations. Primary training seed: 42; seeds 43/44 are stability checks. Omni-MATH remains deferred.", "",
             "Output tokens include reasoning and both answer-extraction outputs. Input processing and the additional teacher-forcing (TF) pass are excluded from this token count. These numbers do not measure end-to-end serving latency.", "",
             "## Main comparison", "",
             "Each method uses the same frozen response bank within a model/dataset. `Selected` uses a dev-selected threshold; `fixed95` uses calibrated confidence 0.95. All learned arms have minimum 2 and cap 16 responses, without continuation.", "",
             "| Model | Evaluation | N | SC16 acc. | ReASC offline acc. / tokens | Mixture selected acc. / tokens | Mixture fixed95 acc. / tokens |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for dataset, split in (("math", "math500"), ("math", "confirm"), ("gsm8k", "test")):
            key = f"{model}_{dataset}/{split}"
            if key not in cells:
                continue
            data = cells[key]; s = data["summary"]
            def value(name: str) -> str:
                v = s[name]
                return f"{v['correct'] * 100:.2f}% / {v['mean_output_tokens']:,.0f}"
            label = "MATH500" if split == "math500" else "MATH confirm500" if split == "confirm" else "GSM8K test"
            lines.append(f"| {DISPLAY[model]} | {label} | {data['questions']} | {s['sc16']['correct'] * 100:.2f}% | {value('reasc_offline')} | {value('mixture_s42__selected')} | {value('mixture_s42__fixed95')} |")
    significant = [(key, control) for key, data in cells.items()
                   for control, v in data["primary_structure_contrasts_seed42"].items()
                   if v["accuracy_difference"] > 0 and v["holm_p_four_primary_structure_contrasts"] < .05]
    conclusion = (f"{len(significant)} primary structure contrast(s) favor mixture at per-cell Holm p < 0.05: "
                  + ", ".join(f"{key} vs {control}" for key, control in significant) + "."
                  if significant else
                  "No per-cell Holm-adjusted primary accuracy contrast currently establishes a semantic-mixture advantage over the four prespecified structure controls. This does not prove equivalence.")
    lines += ["", "MATH500 is the standard OpenAI subset. Confirm500 is a separate original-MATH-test subset excluded from all earlier registered cohorts. `previous_fresh`, reported below for Gemma, is an earlier observed holdout and is exploratory. MATH train/dev use original train questions; the inherited 128-question ReASC calibration cohort contains 79 original-train and 49 original-test questions, disjoint from evaluation cohorts. GSM8K train/dev/calibration come from official train; evaluation uses all 1,319 official test questions.", "",
              conclusion + " Different dev-selected thresholds can change cost; fixed95 and fixed-prefix metrics provide additional comparisons.", "",
              "The common judge is pinned xVerify-3B-Ia; the original ReASC alias was unavailable. This is a controlled reproduction, not a claim of identical historical benchmark scores.", "",
              "## Complete per-evaluation tables", ""]
    for key, data in cells.items():
        lines += [f"### {key}", "", data["evaluation_role"], "",
                  "| Method | Accuracy | Mean responses | Mean output tokens |",
                  "|---|---:|---:|---:|"]
        for name, v in data["summary"].items():
            if name not in BASELINES and "_s42__" not in name:
                continue
            label = name.replace("_s42__", " / ")
            lines.append(f"| {label} | {v['correct'] * 100:.2f}% | {v['samples']:.2f} | {v['mean_output_tokens']:,.0f} |")
        lines += ["", "Primary structure contrasts: mixture selected minus control selected, paired by question.", "",
                  "| Control | Accuracy delta (pp) | Paired 95% CI (pp) | Output-token ratio | Holm p |",
                  "|---|---:|---|---:|---:|"]
        for name, v in data["primary_structure_contrasts_seed42"].items():
            lo, hi = v["accuracy_difference_ci95"]
            lines.append(f"| {name} | {v['accuracy_difference'] * 100:+.2f} | [{lo * 100:+.2f}, {hi * 100:+.2f}] | {v['output_token_ratio']:.3f} | {v['holm_p_four_primary_structure_contrasts']:.4f} |")
        lines += [""]
    lines += ["## Machine-readable results", "",
              "[metrics.json](metrics.json) retains all seeds, common operating points, fixed-prefix metrics, primary/secondary contrasts, source hashes and cost caveats. [provenance/](provenance/) retains cohort identities and original locks. Original artifact paths in those records refer to the local artifact bundle, which is not included in Git.", ""]
    (target / "README.md").write_text("\n".join(lines))
    provenance = target / "provenance"
    provenance.mkdir(exist_ok=True)
    sources = {
        "source_pins.json": run / "source_pins.json",
        "followup_cohorts.json": project / "artifacts/data/pgm_followup/manifest.json",
        "reasc_cohorts.json": project / "artifacts/data/reasc_pgm/manifest.json",
        "excluded_problem_hashes.json": project / "artifacts/data/pgm_asc/excluded_problem_hashes.json",
    }
    for cell in sorted({key.split("/")[0] for key in cells}):
        sources[f"{cell}_selection_lock.json"] = run / cell / "selection_lock.json"
    manifest = {}
    for name, path in sources.items():
        if path.exists():
            shutil.copyfile(path, provenance / name)
            manifest[name] = {"source": str(path.relative_to(project)), "sha256": digest(path)}
    write_json(provenance / "manifest.json", manifest)
    print(f"Exported {len(cells)} completed evaluations to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export(output=args.output)
