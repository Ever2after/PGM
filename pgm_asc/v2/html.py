"""results/v2/report.html: a self-contained page (model, equations, setup, interactive accuracy-token curves, ablations)."""
from __future__ import annotations

import json
from pathlib import Path

from pgm_asc.config import RESULTS
from pgm_asc.v2 import CONFIG
from pgm_asc.v2.model import readable
from pgm_asc.v2.report import DATASET, MODEL, NAMES, cells

TEMPLATE = Path(__file__).with_name("report_template.html")
BASELINES = ["asc", "esc", "reasc_offline", "reasc_online"]


def _r(x, k=4):
    return round(float(x), k)


def cell_payload(r: dict) -> dict:
    sc = r["sc16"]["test"]
    tok = sc["mean_tokens"]
    curves = {m: sorted([[_r(p["mean_tokens"] / tok), _r(100 * p["accuracy"], 2), _r(p["mean_samples"], 2), round(p["mean_tokens"]), p["threshold"]]
                         for p in pts], key=lambda v: v[0]) for m, pts in r["curves"]["test"].items()}
    sel = {}
    for key, rule in [("cf", "cf_accuracy"), ("dev", "accuracy"), ("def", "default")] + \
                     [(f"cf_b{b}", f"cf_budget_{b}") for b in CONFIG["selection"]["budgets"]]:
        sel[key] = {m: [_r(s["test"]["mean_tokens"] / tok), _r(100 * s["test"]["accuracy"], 2), _r(s["test"]["mean_samples"], 2)]
                    for m, s in r["selected"].get(rule, {}).items()}

    def contrasts(rule):
        return {m: [_r(100 * c["accuracy_difference"], 2), *[_r(100 * v, 2) for v in c["accuracy_difference_ci95"]],
                    _r(c["token_ratio"], 3), *[_r(v, 3) for v in c["token_ratio_ci95"]], _r(c["mcnemar_exact_p"], 4)]
                for m, c in r["contrasts"][rule].items()}
    fitted = {a: {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in readable(f["params"]).items()}
              for a, f in r["fitted"].items()}
    return {"key": f"{r['model']}_{r['dataset']}", "model": MODEL.get(r["model"], r["model"]), "dataset": DATASET[r["dataset"]],
            "n": r["questions"]["test"], "n_dev": r["questions"]["dev"], "sc16": [_r(100 * sc["accuracy"], 2), round(tok)],
            "sc16_dev": _r(100 * r["sc16"]["dev"]["accuracy"], 2), "curves": curves, "sel": sel,
            "frontier": {m: _r(100 * v, 2) for m, v in r["frontier_score"].items()},
            "cf": contrasts("cf_accuracy"), "dev": contrasts("accuracy"), "fitted": fitted,
            "ece": {a: _r(c["ece_10bin_all_prefixes"], 3) for a, c in r["calibration"].items()}}


def make_report(path=None, rs=None, extra=()) -> Path:
    path = path or RESULTS / "v2" / "report.html"
    rs = cells() if rs is None else rs
    payload = {"cells": [cell_payload(r) for r in rs], "names": NAMES, "primary": CONFIG["primary"],
               "arms": list(CONFIG["arms"]), "baselines": BASELINES, "extra": list(extra),
               "config": {"lambdas": len(CONFIG["lambdas"]), "gammas": len(CONFIG["gammas"]), "sims": CONFIG["voi"]["simulations"],
                          "horizons": CONFIG["voi"]["horizons"], "grid": CONFIG["grid_size"], "budgets": CONFIG["selection"]["budgets"]}}
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data), encoding="utf-8")
    return path
