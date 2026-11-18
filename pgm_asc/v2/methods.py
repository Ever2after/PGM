"""Evaluate a new stopping/aggregation method under the v2 protocol and draw it on the same figures.

A method is a *threshold family*. For every threshold it decides, per question, how many of the 16 banked
responses to consume and which observed answer to return. Two ways to provide one:

1. Python plugin (replays the bank; recommended)::

       from pgm_asc.v2.methods import register

       @register("my_method", thresholds=[1, 2, 3], default=2, label="My method")
       def my_method(arrays, threshold, context):
           ...                                   # arrays: one split, labels removed (see SPLIT_INPUTS)
           return stop, chosen                   # int arrays [Q]: stop = responses consumed - 1, chosen = answer slot

   then ``pgm v2 method --plugin path/to/file.py`` (or a module name).

2. Decisions CSV (for methods computed elsewhere on the same banks)::

       model,dataset,split,question_id,threshold,samples,response_index
       llama8b,math,test,math_test_0000,0.9,3,0

   ``samples`` = responses consumed (1-16), ``response_index`` = 0-based index (< samples) of a consumed response
   whose extracted answer is returned. Rows for every question of the dev and test splits of each cell, for every
   threshold. Then ``pgm v2 method --csv decisions.csv --name my_method``.

Results go to ``results/v2/methods/<name>/<model>_<dataset>.json``; ``pgm v2 report`` adds every method found there
to the tables, figures and report.html (same axes, next colour in the palette).
"""
from __future__ import annotations

import csv
import importlib
import importlib.util
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from pgm_asc.config import RESULTS
from pgm_asc.utils import log, read_json, utc_now, write_json
from pgm_asc.v2 import CONFIG, result_path
from pgm_asc.v2 import baselines as B
from pgm_asc.v2.data import X_NAMES, load
from pgm_asc.v2.model import Spec
from pgm_asc.v2.run import SPLITS, arm_family, load_traces, score_families

N = 16
# What a method sees for the split it is evaluated on. Labels ("correct", "raw_correct") are removed.
SPLIT_INPUTS = {
    "question_id": "[Q] question ids (cohort order)",
    "ids": "[Q,16] answer slot of each response: exact-string candidates numbered in first-observed order",
    "valid": "[Q,16] the response has a non-empty extracted answer",
    "x": f"[Q,16,{len(X_NAMES)}] per-response features: " + ", ".join(X_NAMES),
    "tokens": "[Q,16] output tokens of each response (body + answer extraction): the cost unit",
    "sim_text": "[Q,16,16] word 3-gram Jaccard similarity of the reasoning texts",
    "sim_hidden": "[Q,16,16] cosine of mean last-layer hidden states (centered on the train mean)",
}
LABELS = ("correct", "raw_correct")


@dataclass
class Method:
    name: str
    fn: Callable
    thresholds: list
    default: object = None
    label: str = ""
    source: str = ""
    per_cell_thresholds: dict = field(default_factory=dict)

    def thresholds_for(self, model, dataset):
        return self.per_cell_thresholds.get((model, dataset), self.thresholds)


REGISTRY: dict[str, Method] = {}


def register(name: str, thresholds, default=None, label: str | None = None):
    """Decorator: register ``fn(arrays, threshold, context) -> (stop, chosen)`` as a method."""
    if name in CONFIG["arms"] or name in B.DEFAULTS:
        raise ValueError(f"{name!r} is already a built-in method name")

    def wrap(fn):
        REGISTRY[name] = Method(name, fn, list(thresholds), default, label or name, getattr(fn, "__module__", ""))
        return fn
    return wrap


def load_plugin(spec: str) -> list[str]:
    """Import a plugin file (``path/to/file.py``) or module; returns the names it registered."""
    before = set(REGISTRY)
    if spec.endswith(".py"):
        path = Path(spec).resolve()
        module_spec = importlib.util.spec_from_file_location(f"pgm_v2_plugin_{path.stem}", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        importlib.import_module(spec)
    added = sorted(set(REGISTRY) - before)
    if not added:
        raise SystemExit(f"{spec}: no method registered (decorate a function with pgm_asc.v2.methods.register)")
    return added


def csv_method(path: str, name: str, label: str | None = None, default=None) -> Method:
    """A method replayed from a decisions CSV (see module docstring)."""
    table = defaultdict(dict)       # (model, dataset, split) -> {threshold: {question_id: (samples, response_index)}}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            thr = _number(row["threshold"])
            table[(row["model"], row["dataset"], row["split"])].setdefault(thr, {})[row["question_id"]] = \
                (int(row["samples"]), int(row["response_index"]))
    if not table:
        raise SystemExit(f"{path}: no rows")

    def fn(arrays, threshold, context):
        rows = table.get((context["model"], context["dataset"], context["split"]), {}).get(threshold)
        if rows is None:
            raise SystemExit(f"{path}: no rows for {context['model']}/{context['dataset']}/{context['split']} threshold {threshold}")
        missing = [q for q in arrays["question_id"] if q not in rows]
        if missing:
            raise SystemExit(f"{path}: {len(missing)} questions missing, e.g. {missing[0]}")
        samples = np.array([rows[q][0] for q in arrays["question_id"]])
        index = np.array([rows[q][1] for q in arrays["question_id"]])
        if (index >= samples).any():
            raise SystemExit(f"{path}: response_index must be < samples")
        return samples - 1, arrays["ids"][np.arange(len(samples)), index].astype(int)

    per_cell = defaultdict(set)
    for (m, d, s), thr in table.items():
        per_cell[(m, d)] |= set(thr)
    thresholds = sorted(set().union(*per_cell.values()))
    return Method(name, fn, thresholds, default, label or name, str(path), {k: sorted(v) for k, v in per_cell.items()})


def _number(text):
    try:
        v = float(text)
        return int(v) if v.is_integer() and "." not in text else v
    except ValueError:
        return text


def family(method: Method, data: dict, split: str, model: str, dataset: str) -> dict:
    """{threshold: (stop, chosen)} on one split, with the method's outputs checked against the protocol."""
    arrays = {k: v for k, v in data[split].items() if k not in LABELS}
    context = {"model": model, "dataset": dataset, "split": split, "train": data["train"], "calibration": data["calibration"],
               "x_names": list(X_NAMES)}
    ids = data[split]["ids"].astype(int)
    rows = np.arange(len(ids))
    out = {}
    for thr in method.thresholds_for(model, dataset):
        stop, chosen = (np.asarray(v).astype(int) for v in method.fn(arrays, thr, context))
        if stop.shape != rows.shape or chosen.shape != rows.shape:
            raise ValueError(f"{method.name}@{thr}: expected arrays of shape ({len(rows)},)")
        if (stop < 0).any() or (stop >= N).any():
            raise ValueError(f"{method.name}@{thr}: stop must be in [0, 15] (responses consumed - 1)")
        seen = (ids == chosen[:, None]) & (np.arange(N)[None, :] <= stop[:, None])
        if not seen.any(1).all():
            raise ValueError(f"{method.name}@{thr}: chosen answer must appear among the consumed responses")
        out[thr] = (stop, chosen)
    return out


def methods_dir(name: str) -> Path:
    return RESULTS / "v2" / "methods" / name


def evaluate(method: Method, models=None, datasets=None) -> list[Path]:
    """Score ``method`` on every cell that has v2 results; the primary PGM arm and SC@k are scored alongside."""
    primary = CONFIG["primary"]
    written = []
    for model in models or CONFIG["models"]:
        for dataset in datasets or CONFIG["datasets"]:
            if not result_path(model, dataset).exists():
                log("v2-method-skip", model=model, dataset=dataset, reason="no results/v2 cell (run `pgm v2 run` first)")
                continue
            data = load(model, dataset)
            traces = load_traces(model, dataset, arms=[primary])
            fams = {}
            for s in SPLITS:
                fams[s] = {"sc": B.sc_family(data[s]), primary: arm_family(traces[(primary, s)], Spec.of(primary)),
                           method.name: family(method, data, s, model, dataset)}
            defaults = {"sc": 16}
            if method.default is not None:
                defaults[method.name] = method.default
            scored = score_families(data, fams, defaults, primary)
            keep = {"sc", primary, method.name}
            result = {"method": method.name, "label": method.label, "source": method.source, "model": model, "dataset": dataset,
                      "created_at": utc_now(), "thresholds": method.thresholds_for(model, dataset), "default": method.default,
                      "sc16": scored["sc16"], "frontier_budgets": scored["frontier_budgets"],
                      "curves": {s: {m: v for m, v in c.items() if m in keep} for s, c in scored["curves"].items()},
                      "selected": {r: {m: v for m, v in sel.items() if m in keep} for r, sel in scored["selected"].items()},
                      "frontier_score": {m: v for m, v in scored["frontier_score"].items() if m in keep},
                      "contrasts": {r: {m: v for m, v in c.items() if m.split("@")[0] == method.name} for r, c in scored["contrasts"].items()}}
            path = methods_dir(method.name) / f"{model}_{dataset}.json"
            write_json(path, result)
            written.append(path)
            sel = result["selected"]["cf_accuracy"][method.name]["test"]
            log("v2-method", method=method.name, model=model, dataset=dataset, frontier=result["frontier_score"][method.name],
                cf_accuracy=sel["accuracy"], cf_tokens=sel["mean_tokens"])
    return written


def extra_results() -> dict:
    """{method name: {(model, dataset): result}} for every evaluated user method."""
    out = {}
    root = RESULTS / "v2" / "methods"
    for path in sorted(root.glob("*/*.json")) if root.exists() else []:
        r = read_json(path)
        out.setdefault(r["method"], {})[(r["model"], r["dataset"])] = r
    return out


def merge_into(cells: list[dict], names: dict, only=None) -> list[str]:
    """Add evaluated user methods to the loaded cell results (in place) so reports treat them like baselines."""
    added = []
    for name, by_cell in extra_results().items():
        if only is not None and name not in only:
            continue
        if not all((r["model"], r["dataset"]) in by_cell for r in cells):
            log("v2-method-partial", method=name, note="not evaluated on every cell; left out of the report")
            continue
        for r in cells:
            e = by_cell[(r["model"], r["dataset"])]
            for s in r["curves"]:
                r["curves"][s][name] = e["curves"][s][name]
            for rule, sel in e["selected"].items():
                if name in sel:
                    r["selected"].setdefault(rule, {})[name] = sel[name]
            r["frontier_score"][name] = e["frontier_score"][name]
            for rule, c in e["contrasts"].items():
                r["contrasts"].setdefault(rule, {}).update(c)
        names[name] = next(iter(by_cell.values()))["label"]
        added.append(name)
    return added
