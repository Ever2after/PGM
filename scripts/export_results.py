"""Publish aggregate metrics and cohort hashes without banks or model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]
MODELS = ("gemma4b", "qwen3b", "qwen7b", "llama8b")
DISPLAY = {"gemma4b": "Gemma-3-4B-IT", "qwen3b": "Qwen2.5-3B-Instruct", "qwen7b": "Qwen2.5-7B-Instruct",
           "llama8b": "Llama-3.1-8B-Instruct"}
BASELINES = ("pass1", "sc16", "asc", "esc", "reasc_offline", "reasc_online", "asc_dev_tuned", "reasc_dev_tuned")
SPLITS = (("math", ("math500", "confirm", "previous_fresh")), ("gsm8k", ("test",)),
          ("omnimath_indomain", ("test",)), ("omnimath", ("test",)))
# Reported sections. Other exported evaluations (MATH confirm500, Gemma previous_fresh) stay in
# metrics.json but are excluded from the tables.
SECTIONS = (
    ("In-domain evaluation",
     "Heads, calibrators and thresholds are fitted on the same dataset's train/dev/calibration cohorts.",
     (("math", "math500", "MATH500"), ("gsm8k", "test", "GSM8K test"), ("omnimath_indomain", "test", "Omni-MATH test"))),
    ("Cross-dataset transfer",
     "Each model's locked MATH cell (heads, calibrators, thresholds, baseline offline calibration) is applied unchanged "
     "to Omni-MATH, with no Omni labels. It replays the same Omni test responses as the in-domain Omni cell.",
     (("omnimath", "test", "MATH → Omni-MATH test"),)),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def export(project: Path = PROJECT, output: Path | None = None) -> None:
    project = Path(project)
    target = Path(output) if output is not None else project / "results" / "followup"
    run = project / "artifacts" / "runs" / "pgm_followup"
    # Evaluations already exported from another artifact bundle are kept when
    # their source is absent here, so partial bundles can extend the export.
    previous = target / "metrics.json"
    retained = json.loads(previous.read_text())["evaluations"] if previous.exists() else {}
    cells, summaries = {}, {}
    for model in MODELS:
        for dataset, splits in SPLITS:
            for split in splits:
                key = f"{model}_{dataset}/{split}"
                source = Path("artifacts/runs/pgm_followup") / key / "study_evaluation.json"
                if (project / source).exists():
                    data = json.loads((project / source).read_text())
                    # Retain all three training seeds, common operating points and statistics.
                    # Per-question decisions, generated text and threshold sweeps stay local.
                    summaries[key] = {k: data[k] for k in (
                        "created_at", "model", "dataset", "split", "questions", "evaluation_role", "summary",
                        "primary_structure_contrasts_seed42", "secondary_baseline_contrasts_seed42", "seed_stability",
                        "fixed_prefix_metrics", "supervision_budget", "inference_cost_note", "multiple_testing_note")}
                    summaries[key].update(source=str(source), source_sha256=digest(project / source))
                elif key in retained:
                    summaries[key] = retained[key]
                if key in summaries:
                    cells[key] = summaries[key]
    if not any((project / s["source"]).exists() for s in summaries.values()):
        raise SystemExit("No completed study_evaluation.json files found; existing exports left untouched")
    target.mkdir(parents=True, exist_ok=True)
    reported = {f"{m}_{d}/{s}": (title, label) for title, _, rows in SECTIONS for d, s, label in rows for m in MODELS}
    omni_status = "completed" if all(f"{m}_omnimath/test" in cells for m in MODELS) else "partial"
    write_json(target / "metrics.json", {"schema_version": 1, "primary_seed": 42, "omni_status": omni_status,
                                         "reported_sections": {k: v[0] for k, v in reported.items() if k in cells},
                                         "evaluations": summaries})
    lines = ["# Follow-up results", "",
             "Aggregate results exported from the completed, locked evaluations. Primary training seed: 42; seeds 43/44 are stability checks. "
             "In-domain results and cross-dataset transfer are reported separately.", "",
             "Output tokens include reasoning and both answer-extraction outputs. Input processing and the additional teacher-forcing (TF) pass are excluded from this token count. These numbers do not measure end-to-end serving latency.", "",
             "Each method uses the same frozen response bank within a model/dataset. `Selected` uses a dev-selected threshold; `fixed95` uses calibrated confidence 0.95. All learned arms have minimum 2 and cap 16 responses, without continuation.", ""]
    header = ["| Model | Evaluation | N | SC16 acc. | ReASC offline acc. / tokens | ASC dev-tuned acc. / tokens | Mixture selected acc. / tokens | Mixture fixed95 acc. / tokens |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for title, description, rows in SECTIONS:
        lines += [f"## {title}", "", description, ""] + header
        for model in MODELS:
            for dataset, split, label in rows:
                key = f"{model}_{dataset}/{split}"
                if key not in cells:
                    continue
                data = cells[key]; s = data["summary"]
                value = lambda name: f"{s[name]['correct'] * 100:.2f}% / {s[name]['mean_output_tokens']:,.0f}"
                lines.append(f"| {DISPLAY[model]} | {label} | {data['questions']} | {s['sc16']['correct'] * 100:.2f}% | {value('reasc_offline')} | "
                             f"{value('asc_dev_tuned')} | {value('mixture_s42__selected')} | {value('mixture_s42__fixed95')} |")
        lines.append("")
    shown = {k: v for k, v in cells.items() if k in reported}
    significant = [(key, control) for key, data in shown.items()
                   for control, v in data["primary_structure_contrasts_seed42"].items()
                   if v["accuracy_difference"] > 0 and v["holm_p_four_primary_structure_contrasts"] < .05]
    conclusion = (f"{len(significant)} of {4 * len(shown)} reported primary structure contrasts favor mixture at per-cell Holm p < 0.05: "
                  + ", ".join(f"{key} vs {control}" for key, control in significant) + ". "
                  if significant else
                  "No reported per-cell Holm-adjusted primary accuracy contrast establishes a semantic-mixture advantage over the four prespecified structure controls. ")
    lines += ["## Cohorts and caveats", "",
              "MATH500 is the standard OpenAI subset; MATH train/dev use original train questions and the inherited 128-question ReASC calibration cohort (79 original-train, 49 original-test questions, disjoint from evaluation). "
              "GSM8K train/dev/calibration come from official train; evaluation uses all 1,319 official test questions. "
              "Omni-MATH evaluates 999 distinct questions from the 1,000-row ReASC archive with a 2,048-token body cap. Omni-MATH has no train split, so in-domain calibration/train/dev (128/800/200) are drawn from the rest of the full Omni-MATH release with every test question removed by normalized problem identity. "
              "MATH confirm500 and the exploratory Gemma previous_fresh subset remain in metrics.json but are excluded from these tables.", "",
              conclusion + "Different dev-selected thresholds can change cost, so accuracy contrasts are not equal-budget comparisons; fixed95 and fixed-prefix metrics provide additional comparisons.", "",
              "Llama-3.1-8B-Instruct ships without a pad token; the TF worker fills it with EOS (masked, output-neutral). Its Omni-MATH generation ran with a 0.5 vLLM memory fraction instead of 0.25, because the smaller KV cache triggered a vLLM 0.11 preemption assertion; sampling and seeds are unchanged.", "",
              "The common judge is pinned xVerify-3B-Ia; the original ReASC alias was unavailable. This is a controlled reproduction, not a claim of identical historical benchmark scores.", "",
              "## Complete per-evaluation tables", ""]
    for title, _, rows in SECTIONS:
        lines += [f"### {title}", ""]
        for model in MODELS:
            for dataset, split, label in rows:
                key = f"{model}_{dataset}/{split}"
                if key not in cells:
                    continue
                data = cells[key]
                lines += [f"#### {DISPLAY[model]}, {label}", "", data["evaluation_role"], "",
                          "| Method | Accuracy | Mean responses | Mean output tokens |",
                          "|---|---:|---:|---:|"]
                for name, v in data["summary"].items():
                    if name not in BASELINES and "_s42__" not in name:
                        continue
                    lines.append(f"| {name.replace('_s42__', ' / ')} | {v['correct'] * 100:.2f}% | {v['samples']:.2f} | {v['mean_output_tokens']:,.0f} |")
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
        "omnimath_indomain_cohorts.json": project / "artifacts/data/pgm_followup/omnimath_indomain/manifest.json",
        "llama8b_source_pin.json": run / "llama8b_source_pin.json",
    }
    for cell in sorted({key.split("/")[0] for key in cells}):
        sources[f"{cell}_selection_lock.json"] = run / cell / "selection_lock.json"
    previous = provenance / "manifest.json"
    manifest = json.loads(previous.read_text()) if previous.exists() else {}
    for name, path in sources.items():
        if path.exists():
            shutil.copyfile(path, provenance / name)
            manifest[name] = {"source": str(path.relative_to(project)), "sha256": digest(path)}
    write_json(provenance / "manifest.json", manifest)
    print(f"Exported {len(cells)} completed evaluations to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT,
                        help="root holding artifacts/ (e.g. an extracted artifact bundle)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export(project=args.project, output=args.output or PROJECT / "results" / "followup")
