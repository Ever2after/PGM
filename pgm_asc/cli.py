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
    print("Wrote results/v1/README.md, summary.csv, figures/, report.html")


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


def cmd_v2(a):
    from pgm_asc.v2 import CONFIG
    models, datasets = a.models or CONFIG["models"], a.datasets or CONFIG["datasets"]
    if a.stage == "prepare":
        from multiprocessing import get_context
        from pgm_asc.v2 import data_path
        from pgm_asc.v2.data import prepare
        todo = [(m, d) for m in models for d in datasets if a.overwrite or not data_path(m, d).exists()]
        if todo:
            with get_context("spawn").Pool(min(a.workers, len(todo))) as pool:
                pool.starmap(prepare, todo)
    elif a.stage == "run":
        from pgm_asc.v2.run import run
        run(models, datasets, a.workers)
    elif a.stage == "rescore":
        from pgm_asc.v2.run import rescore
        for m in models:
            for d in datasets:
                rescore(m, d)
    elif a.stage == "method":
        from pgm_asc.v2.methods import REGISTRY, csv_method, evaluate, load_plugin
        if bool(a.plugin) == bool(a.csv):
            raise SystemExit("give exactly one of --plugin FILE_OR_MODULE or --csv FILE")
        if a.plugin:
            names = load_plugin(a.plugin)
            methods = [REGISTRY[n] for n in names if a.name is None or n == a.name]
        else:
            if not a.name:
                raise SystemExit("--csv needs --name")
            methods = [csv_method(a.csv, a.name, a.label)]
        for method in methods:
            paths = evaluate(method, a.models, a.datasets)
            print(f"{method.name}: {len(paths)} cells -> results/v2/methods/{method.name}/")
    elif a.stage == "plot":
        from pgm_asc.v2.report import plot
        if not a.methods or not a.out:
            raise SystemExit("plot needs --methods and --out")
        path = plot(a.methods, a.out, a.cells, a.title or "accuracy vs output tokens (test)")
        print(f"Wrote {path.with_suffix('.png')} and {path.with_suffix('.svg')}")
    else:
        from pgm_asc.v2.report import report
        report(a.methods)


def cmd_p1_study(a):
    from importlib import import_module
    from pgm_asc.utils import read_json
    study = read_json(a.config) if a.config else None
    result = import_module(f"pgm_asc.{a.study_module}.run").run(
        a.models, a.datasets, a.workers, a.out, a.stage, a.overwrite, study)
    print(f"{a.command} {a.stage} complete: {result}")


def cmd_p1_stop(a):
    if a.stage == "trace":
        from pgm_asc.p1_stopping.run import run
        result = run(a.models, a.datasets, a.workers, a.overwrite)
    else:
        from pgm_asc.p1_stopping.report import report
        result = report()
    print(f"P1 stopping {a.stage} complete: {result}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pgm", description="PGM adaptive self-consistency: data, fitting, evaluation, reports.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="which banks, policies and evaluations are present").set_defaults(run=cmd_status)

    p = sub.add_parser("download", help="fetch released artifacts from the Hugging Face Hub")
    p.add_argument("--only", nargs="+", default=["cohorts", "banks", "policies", "decisions", "v2"],
                   choices=["cohorts", "banks", "policies", "decisions", "v2", "logprobs"],
                   help="logprobs (per-token top-20 log-probabilities) is not downloaded unless listed")
    p.add_argument("--models", nargs="+", choices=list(MODELS))
    p.add_argument("--datasets", nargs="+", choices=DATASETS)
    p.add_argument("--no-embeddings", action="store_true", help="skip the embedding matrices (needed for fitting and evaluation; skip only to inspect responses)")
    p.add_argument("--revision")
    p.set_defaults(run=cmd_download)

    sub.add_parser("verify", help="check local artifacts against manifest.json").set_defaults(run=cmd_verify)

    p = sub.add_parser("fit", help="(v1) train all partition-mixture heads of a cell and select thresholds on dev (CPU)")
    p.add_argument("cell", choices=[c.name for c in cells() if not c.transfer])
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(run=cmd_fit)

    p = sub.add_parser("evaluate", help="(v1) replay the saved policy and all baselines on the test bank (CPU)")
    p.add_argument("cell", choices=CELL_NAMES + ["all"])
    p.set_defaults(run=cmd_evaluate)

    sub.add_parser("report", help="(v1) tables, figures and report.html from results/v1/evaluations").set_defaults(run=cmd_report)
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

    p = sub.add_parser("v2", help="PGM v2 (main study): prepare, run, rescore, method, plot, report (CPU)",
                       description="prepare: compact inputs from the banks | run: fit + replay every arm and baseline | "
                                   "rescore: re-run the evaluation protocol from saved traces | method: evaluate a new method "
                                   "(--plugin or --csv) | plot: one accuracy-token figure | report: tables, figures, report.html")
    p.add_argument("stage", choices=["prepare", "run", "rescore", "method", "plot", "report"])
    p.add_argument("--models", nargs="+", choices=list(MODELS))
    p.add_argument("--datasets", nargs="+", choices=DATASETS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true", help="prepare: rebuild existing .npz inputs")
    p.add_argument("--plugin", help="method: Python file or module that registers methods with pgm_asc.v2.methods.register")
    p.add_argument("--csv", help="method: decisions CSV (model,dataset,split,question_id,threshold,samples,response_index)")
    p.add_argument("--name", help="method: name for a --csv method (or pick one method from a plugin)")
    p.add_argument("--label", help="method: display label for a --csv method")
    p.add_argument("--methods", nargs="+", help="plot: methods to draw | report: user methods to include (default: all evaluated)")
    p.add_argument("--cells", nargs="+", help="plot: cells such as llama8b_math (default: all)")
    p.add_argument("--out", help="plot: output path without extension")
    p.add_argument("--title", help="plot: figure title")
    p.set_defaults(run=cmd_v2)

    studies = {
        "hierarchical": ("hierarchical", "hierarchical_crp", "P1 hierarchical CRP structure ablation"),
        "p1-objective": ("p1_objectives", "p1_objectives", "P1 joint/prefix/mixed objective ablation"),
        "p1-predictive": ("p1_predictive", "p1_predictive", "P1 Mixed plus next-answer NLL"),
        "p1-transition": ("p1_transitions", "p1_transitions", "P1 Mixed plus repair/damage transition NLL"),
        "p1-brier": ("p1_brier", "p1_brier", "P1 Mixed plus repair/damage Brier loss"),
    }
    for name, (module, output, description) in studies.items():
        p = sub.add_parser(name, help=description)
        p.add_argument("stage", choices=["fit", "run", "report"])
        p.add_argument("--models", nargs="+", choices=list(MODELS))
        p.add_argument("--datasets", nargs="+", choices=DATASETS)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--out", help=f"default: results/{output}")
        p.add_argument("--config", help=f"default: configs/{module}.json")
        p.add_argument("--overwrite", action="store_true")
        p.set_defaults(run=cmd_p1_study, study_module=module)

    p = sub.add_parser("p1-stop", help="frozen P1 heads with the standard v2 stopping protocol")
    p.add_argument("stage", choices=["trace", "report"])
    p.add_argument("--models", nargs="+", choices=list(MODELS))
    p.add_argument("--datasets", nargs="+", choices=DATASETS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(run=cmd_p1_stop)

    args = parser.parse_args(argv)
    if args.command in (*studies, "p1-stop") and args.workers < 1:
        parser.error("workers must be positive")
    if args.command == "collect" and args.stage in ("generate", "features", "judge") and not args.split:
        parser.error("--split is required for this stage")
    args.run(args)


if __name__ == "__main__":
    main()
