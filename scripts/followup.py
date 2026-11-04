"""One entry point for inspecting, fitting, replaying and exporting the follow-up."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
CONFIGS = PROJECT / "configs" / "pgm_followup"
MODELS = ("gemma4b", "qwen3b", "qwen7b", "llama8b")
# In-domain cells are fitted on their own dataset.
CELLS = tuple(f"{model}_{dataset}" for model in MODELS for dataset in ("math", "gsm8k", "omnimath_indomain"))
# Transfer cells copy the locked MATH cell and replay the in-domain Omni test bank; `fit` writes their transfer lock.
OMNI_CELLS = tuple(f"{model}_omnimath" for model in MODELS)


def status() -> None:
    print("| Cell | Locked | Completed evaluation splits |")
    print("|---|---|---|")
    for cell in CELLS + OMNI_CELLS:
        cfg = json.loads((CONFIGS / f"{cell}.json").read_text())
        root = PROJECT / cfg["run_dir"]
        complete = [split for split in cfg["evaluation_splits"]
                    if (root / split / "study_evaluation.json").exists()]
        print(f"| {cell} | {(root / 'selection_lock.json').exists()} | {', '.join(complete) or 'pending'} |")
    print("\nGPU collection plus fit/evaluate for one cell: scripts/run_cell.py --config <cell config>.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="inspect local artifacts; no GPU or dependencies required")
    exporter = sub.add_parser("export", help="export completed aggregate results for GitHub")
    exporter.add_argument("--project", type=Path, default=PROJECT, help="root holding artifacts/")
    for command in ("fit", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--cell", choices=CELLS + OMNI_CELLS, required=True)
        if command == "evaluate":
            child.add_argument("--split", required=True, choices=("math500", "confirm", "previous_fresh", "test"))
    args = parser.parse_args()
    os.chdir(PROJECT)
    if args.command == "status":
        status()
        return
    if args.command == "export":
        from export_results import export
        export(project=args.project, output=PROJECT / "results" / "followup")
        return
    from pgm_asc.core.io import read_json
    path = f"configs/pgm_followup/{args.cell}.json"
    cfg = read_json(path)
    if args.command == "fit":
        from pgm_asc.experiments.study_fit import ready, select
        if not ready(cfg):
            parser.error("Fitting banks are incomplete. Restore/collect train, dev and calibration first.")
        select(cfg, path)
    else:
        if args.split not in cfg["evaluation_splits"]:
            parser.error(f"{args.cell} has no evaluation split {args.split!r}")
        from pgm_asc.evaluation.study import evaluate
        evaluate(cfg, args.split)


if __name__ == "__main__":
    main()
