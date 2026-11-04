"""Command line: ``pgm <command>`` (or ``python -m pgm_asc <command>``)."""
from __future__ import annotations

import argparse

from pgm_asc.config import MODELS, STUDY, cells

DATASETS = list(STUDY["datasets"])
CELL_NAMES = [c.name for c in cells()]


def cmd_status(_):
    from pgm_asc.config import bank_path, cohort_path, decisions_path, evaluation_path, policy_path
    print(f"{'cell':26s} {'bank':5s} {'policy':7s} {'evaluated':9s} decisions")
    for c in cells():
        bank = bank_path(c.model, c.eval_dataset).exists() and cohort_path(c.eval_dataset).exists()
        print(f"{c.name:26s} {'yes' if bank else '-':5s} {'yes' if policy_path(c.policy_name).exists() else '-':7s} "
              f"{'yes' if evaluation_path(c.name).exists() else '-':9s} {'yes' if decisions_path(c.name).exists() else '-'}")


def cmd_download(a):
    from pgm_asc.hub import HUB_DATASET, download
    present, total = download(a.only, a.models, a.datasets, not a.no_embeddings, a.revision)
    print(f"{HUB_DATASET}: {present}/{total} released files present and verified")


def cmd_verify(_):
    from pgm_asc.hub import verify
    present, total = verify(missing_ok=True)
    print(f"{present}/{total} released files present; all present files match manifest.json")


def cmd_fit(a):
    from pgm_asc.config import parse_cell
    from pgm_asc.fit import fit_cell
    fit_cell(parse_cell(a.cell), overwrite=a.overwrite)


def cmd_evaluate(a):
    from pgm_asc.config import parse_cell
    from pgm_asc.evaluate import evaluate_cell
    for name in (CELL_NAMES if a.cell == "all" else [a.cell]):
        evaluate_cell(parse_cell(name))


def cmd_report(_):
    from pgm_asc.report.figures import make_figures
    from pgm_asc.report.html import make_report
    from pgm_asc.report.tables import export
    export()
    make_figures()
    make_report()
    print("Wrote results/README.md, results/summary.csv, results/figures/, results/report.html")


def cmd_manifest(_):
    from pgm_asc.hub import build_manifest
    print(f"manifest.json: {len(build_manifest())} files")


def cmd_collect(a):
    kwargs = {"limit": a.limit, "memory": a.memory}
    if a.stage == "run":
        from pgm_asc.collect.run import run_pipeline
        run_pipeline(a.model, a.dataset, tuple(int(g) for g in a.gpus.split(",")), a.memory, a.limit)
    elif a.stage == "pack":
        from pgm_asc.collect.pack import pack
        pack(a.model, a.dataset)
    elif a.stage == "generate":
        from pgm_asc.collect.generate import generate
        generate(a.model, a.dataset, a.split, a.shard, a.shards, allow_test=a.allow_test, **kwargs)
    elif a.stage == "features":
        from pgm_asc.collect.features import features
        features(a.model, a.dataset, a.split, a.follow, allow_test=a.allow_test, **kwargs)
    elif a.stage == "judge":
        from pgm_asc.collect.judge import judge
        judge(a.model, a.dataset, a.split, a.follow, allow_test=a.allow_test, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pgm", description="PGM adaptive self-consistency: data, fitting, evaluation, reports.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="which banks, policies and evaluations are present").set_defaults(run=cmd_status)

    p = sub.add_parser("download", help="fetch released artifacts from the Hugging Face Hub")
    p.add_argument("--only", nargs="+", default=["cohorts", "banks", "policies", "decisions"],
                   choices=["cohorts", "banks", "policies", "decisions", "logprobs"],
                   help="logprobs (per-token top-20 log-probabilities) is not downloaded unless listed")
    p.add_argument("--models", nargs="+", choices=list(MODELS))
    p.add_argument("--datasets", nargs="+", choices=DATASETS)
    p.add_argument("--no-embeddings", action="store_true", help="skip the embedding matrices (needed for fitting and evaluation; skip only to inspect responses)")
    p.add_argument("--revision")
    p.set_defaults(run=cmd_download)

    sub.add_parser("verify", help="check local artifacts against manifest.json").set_defaults(run=cmd_verify)

    p = sub.add_parser("fit", help="train all heads of a cell and select thresholds on dev (CPU)")
    p.add_argument("cell", choices=[c.name for c in cells() if not c.transfer])
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(run=cmd_fit)

    p = sub.add_parser("evaluate", help="replay the saved policy and all baselines on the test bank (CPU)")
    p.add_argument("cell", choices=CELL_NAMES + ["all"])
    p.set_defaults(run=cmd_evaluate)

    sub.add_parser("report", help="tables, figures and report.html from results/evaluations").set_defaults(run=cmd_report)
    sub.add_parser("manifest", help="(maintainers) hash artifacts into manifest.json before uploading").set_defaults(run=cmd_manifest)

    p = sub.add_parser("collect", help="build a new response bank on GPUs")
    p.add_argument("stage", choices=["run", "generate", "features", "judge", "pack"])
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--split", choices=["calibration", "train", "dev", "test"])
    p.add_argument("--gpus", default="0,1,2,3", help="run: generation, generation, teacher forcing, judge")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=2)
    p.add_argument("--memory", type=float, default=0.25, help="GPU memory fraction per worker (0.5 for Llama-3.1-8B on Omni-MATH)")
    p.add_argument("--limit", type=int, help="smoke test: first N questions only")
    p.add_argument("--follow", action="store_true", help="keep polling for new responses until the split is complete")
    p.add_argument("--allow-test", action="store_true", help="collect a test bank before any policy is fitted")
    p.set_defaults(run=cmd_collect)

    args = parser.parse_args(argv)
    if args.command == "collect" and args.stage in ("generate", "features", "judge") and not args.split:
        parser.error("--split is required for this stage")
    args.run(args)


if __name__ == "__main__":
    main()
