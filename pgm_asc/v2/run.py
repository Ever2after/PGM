"""Fit every arm on train, replay dev and test, select operating points on dev, compare on test.

Outputs per cell (model x dataset):
* results/v2/cells/<cell>.json   fitted parameters, threshold curves (dev and test) of every arm and baseline,
                                 operating points selected on dev and by 2-fold cross-fitting on test, paired contrasts
* artifacts/v2/traces/<cell>.npz per-prefix posterior traces (chosen, confidence, OTHER, VOI) of every arm
"""
from __future__ import annotations

import time
from multiprocessing import get_context

import numpy as np
from scipy.stats import binomtest

from pgm_asc.config import ARTIFACTS
from pgm_asc.utils import log, utc_now, write_json
from pgm_asc.v2 import CONFIG, OTHER, result_path
from pgm_asc.v2 import baselines as B
from pgm_asc.v2.data import X_NAMES, load
from pgm_asc.v2.model import Spec, fit, from_json, posterior, to_json, transforms
from pgm_asc.v2.voi import voi

SPLITS = (CONFIG["select_split"], CONFIG["eval_split"])


def x_index():
    return [X_NAMES.index(n) for n in CONFIG["x_features"]]


def traces_path(model, dataset):
    return ARTIFACTS / "v2" / "traces" / f"{model}_{dataset}.npz"


# ------------------------------------------------------------------ worker tasks
def fit_task(args):
    model, dataset, fit_as = args
    started = time.perf_counter()
    data = load(model, dataset)
    spec = Spec.of(fit_as)
    tf = transforms(data[CONFIG["fit_split"]], spec, x_index())
    params, summary = fit(data[CONFIG["fit_split"]], spec, tf)
    summary["seconds"] = time.perf_counter() - started
    log("v2-fit", model=model, dataset=dataset, arm=fit_as, nll=summary["nll_per_question"], seconds=summary["seconds"])
    return (model, dataset, fit_as), {"params": to_json(params), "transforms": tf, "fit": summary}


def trace_task(args):
    model, dataset, arm, split, fitted = args
    started = time.perf_counter()
    data = load(model, dataset)
    spec = Spec.of(arm)
    params = from_json(fitted["params"])
    p = posterior(params, spec, data[split], fitted["transforms"], keep_joint=spec.stop == "voi")
    out = {"chosen": p.chosen.astype(np.int8), "confidence": p.confidence.astype(np.float32),
           "other": p.post[:, :, OTHER].astype(np.float32)}
    if spec.stop == "voi":
        out["voi"] = voi(params, spec, data[split], p, key=f"{arm}/{split}").astype(np.float32)
    log("v2-trace", model=model, dataset=dataset, arm=arm, split=split, seconds=time.perf_counter() - started)
    return (model, dataset, arm, split), out


# ------------------------------------------------------------------ replay
def arm_family(tr: dict, spec: Spec) -> dict:
    """{threshold: (stop index, chosen slot)} for a fitted arm."""
    minimum, cap = CONFIG["stopping"]["minimum"], CONFIG["stopping"]["cap"]
    chosen = tr["chosen"].astype(int)
    t = np.arange(1, cap + 1)
    base = (t >= minimum)[None, :] & (chosen >= 0)
    out = {}
    if spec.stop == "confidence":
        beats = tr["confidence"] > tr["other"] if spec.other else True
        for gamma in CONFIG["gammas"]:
            ok = base & beats & (tr["confidence"] >= gamma)
            ok[:, cap - 1] = True
            out[gamma] = ok.argmax(1)
    else:
        v = np.concatenate([tr["voi"], np.zeros((len(chosen), 1))], 1)
        for lam in CONFIG["lambdas"]:
            ok = base & (v < lam)
            ok[:, cap - 1] = True
            out[lam] = ok.argmax(1)
    rows = np.arange(len(chosen))
    return {k: (s, np.maximum(chosen[rows, s], 0)) for k, s in out.items()}


def outcome(arrays, stop, chosen):
    rows = np.arange(len(stop))
    cum = np.cumsum(arrays["tokens"], 1)
    return arrays["correct"][rows, chosen], cum[rows, stop], stop + 1


def subset(arrays, rows):
    return {k: arrays[k][rows] for k in ("correct", "tokens")}


def curve(arrays, family):
    points = []
    for thr, (stop, chosen) in family.items():
        c, tok, n = outcome(arrays, stop, chosen)
        points.append({"threshold": thr, "accuracy": float(c.mean()), "mean_tokens": float(tok.mean()), "mean_samples": float(n.mean())})
    return points


def select(points, rule, sc16):
    """Dev operating point: 'accuracy' = fewest tokens with acc >= SC16 - tol; 'budget_b' = best acc with tokens <= b*SC16."""
    if rule == "accuracy":
        ok = [p for p in points if p["accuracy"] >= sc16["accuracy"] - CONFIG["selection"]["accuracy_tolerance"] - 1e-12]
        return min(ok, key=lambda p: (p["mean_tokens"], -p["accuracy"])) if ok else max(points, key=lambda p: (p["accuracy"], -p["mean_tokens"]))
    b = float(rule.split("_")[1])
    ok = [p for p in points if p["mean_tokens"] <= b * sc16["mean_tokens"] + 1e-9]
    return max(ok, key=lambda p: (p["accuracy"], -p["mean_tokens"])) if ok else min(points, key=lambda p: p["mean_tokens"])


def paired(a_correct, a_tokens, b_correct, b_tokens, replicates, seed):
    a, b = a_correct.astype(int), b_correct.astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (replicates, len(a)))
    delta = (a[idx] - b[idx]).mean(1)
    ratio = a_tokens[idx].mean(1) / b_tokens[idx].mean(1)
    win, lose = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return {"accuracy_difference": float((a - b).mean()), "accuracy_difference_ci95": np.quantile(delta, [.025, .975]).tolist(),
            "token_ratio": float(a_tokens.mean() / b_tokens.mean()), "token_ratio_ci95": np.quantile(ratio, [.025, .975]).tolist(),
            "only_a_correct": win, "only_b_correct": lose,
            "mcnemar_exact_p": float(binomtest(win, win + lose, .5).pvalue) if win + lose else 1.}


def envelope_score(points, anchor, budgets):
    """Mean over budgets of the best accuracy reachable within the budget (test-set frontier; SC@1 anchor)."""
    pts = points + [anchor]
    return float(np.mean([max([p["accuracy"] for p in pts if p["mean_tokens"] <= b + 1e-9] or [0.]) for b in budgets]))


def score_families(data: dict, fams: dict, defaults: dict, primary: str) -> dict:
    """The evaluation protocol for any set of threshold families (PGM arms, baselines, user methods).

    ``fams[split][method] = {threshold: (stop index [Q], chosen slot [Q])}`` for the dev and test splits; it must
    contain "sc" (SC@k, the reference) and ``primary``. Returns curves, operating points (dev-selected, 2-fold
    cross-fitted on test, published defaults), paired contrasts of ``primary`` against every other method, and the
    frontier score, all on test.
    """
    dev_split, test_split = SPLITS
    curves = {s: {m: curve(data[s], f) for m, f in fams[s].items()} for s in SPLITS}
    sc16 = {s: next(p for p in curves[s]["sc"] if p["threshold"] == 16) for s in SPLITS}
    rules = ["accuracy"] + [f"budget_{b}" for b in CONFIG["selection"]["budgets"]]
    selected, per_question = {}, {}
    for rule in rules:
        selected[rule] = {}
        for m in fams[test_split]:
            point = select(curves[dev_split][m], rule, sc16[dev_split])
            stop, chosen = fams[test_split][m][point["threshold"]]
            c, tok, n = outcome(data[test_split], stop, chosen)
            selected[rule][m] = {"threshold": point["threshold"], "dev": point,
                                 "test": {"accuracy": float(c.mean()), "mean_tokens": float(tok.mean()), "mean_samples": float(n.mean())}}
            per_question[(rule, m)] = (c, tok)
    # Cross-fitting on test: thresholds chosen on one half of the test questions, applied to the other half.
    # Removes the dev/test difficulty shift from the selection while never using a question's own label.
    cf = CONFIG["crossfit"]
    nq = len(data[test_split]["ids"])
    fold = np.random.default_rng(cf["seed"]).permutation(nq) % cf["folds"]
    for rule in rules:
        selected[f"cf_{rule}"] = {}
        for m, fam in fams[test_split].items():
            c_all, tok_all, n_all, thresholds = np.zeros(nq, bool), np.zeros(nq), np.zeros(nq), []
            for k in range(cf["folds"]):
                pick, held = fold == k, fold != k
                part = subset(data[test_split], pick)
                points = curve(part, {t: (st[pick], ch[pick]) for t, (st, ch) in fam.items()})
                sc_part = curve(part, {16: tuple(v[pick] for v in fams[test_split]["sc"][16])})[0]
                point = select(points, rule, sc_part)
                stop, chosen = fam[point["threshold"]]
                c, tok, n = outcome(data[test_split], stop, chosen)
                c_all[held], tok_all[held], n_all[held] = c[held], tok[held], n[held]
                thresholds.append(point["threshold"])
            selected[f"cf_{rule}"][m] = {"threshold": thresholds, "test": {"accuracy": float(c_all.mean()),
                                         "mean_tokens": float(tok_all.mean()), "mean_samples": float(n_all.mean())}}
            per_question[(f"cf_{rule}", m)] = (c_all, tok_all)
    for m, thr in defaults.items():
        stop, chosen = fams[test_split][m][thr]
        c, tok, n = outcome(data[test_split], stop, chosen)
        selected.setdefault("default", {})[m] = {"threshold": thr, "test": {"accuracy": float(c.mean()), "mean_tokens": float(tok.mean()),
                                                                          "mean_samples": float(n.mean())}}
        per_question[("default", m)] = (c, tok)
    bs = CONFIG["bootstrap"]
    contrasts = {}
    for rule in rules + [f"cf_{r}" for r in rules]:
        a = per_question[(rule, primary)]
        contrasts[rule] = {m: paired(*a, *per_question[(rule, m)], bs["replicates"], bs["seed"]) for m in fams[test_split] if m != primary}
        contrasts[rule].update({f"{m}@default": paired(*a, *per_question[("default", m)], bs["replicates"], bs["seed"]) for m in defaults})
    anchor = next(p for p in curves[test_split]["sc"] if p["threshold"] == 1)
    budgets = np.linspace(next(p for p in curves[test_split]["sc"] if p["threshold"] == 2)["mean_tokens"], sc16[test_split]["mean_tokens"], 60)
    frontier = {m: envelope_score(pts, anchor, budgets) for m, pts in curves[test_split].items()}
    return {"sc16": sc16, "curves": curves, "selected": selected, "contrasts": contrasts, "frontier_score": frontier,
            "frontier_budgets": [float(budgets[0]), float(budgets[-1])]}


def assemble(model, dataset, data, fitted, traces, seconds) -> dict:
    arms = list(CONFIG["arms"])
    test_split = SPLITS[1]
    fams = {s: B.families(data, s) for s in SPLITS}
    for arm in arms:
        spec = Spec.of(arm)
        for s in SPLITS:
            fams[s][arm] = arm_family(traces[(arm, s)], spec)
    scored = score_families(data, fams, B.DEFAULTS, CONFIG["primary"])
    calibration = {}
    for arm in arms:
        tr = traces[(arm, test_split)]
        ok = tr["chosen"] >= 0
        conf = tr["confidence"][ok]
        hit = data[test_split]["correct"][np.nonzero(ok)[0], tr["chosen"][ok]]
        bins = np.minimum((conf * 10).astype(int), 9)
        ece = sum(abs(hit[bins == k].mean() - conf[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any())
        calibration[arm] = {"ece_10bin_all_prefixes": float(ece), "mean_confidence": float(conf.mean()), "accuracy": float(hit.mean())}
    return {"model": model, "dataset": dataset, "created_at": utc_now(), "seconds": seconds,
            "questions": {s: int(len(data[s]["ids"])) for s in ("train", *SPLITS)}, "config": CONFIG,
            "fitted": {k: v for k, v in fitted.items()}, **scored, "calibration": calibration,
            "notes": {"cost": "Output tokens = generated tokens + both answer-extraction outputs (v1 definition).",
                      "selection": "Thresholds are selected on dev only; test is replayed once per selected threshold.",
                      "frontier_score": "Mean, over 60 budgets between SC@2 and SC16 test tokens, of the best test accuracy "
                                        "within budget over the method's threshold family (an oracle summary applied equally to every method)."}}


def run(models=None, datasets=None, workers=8):
    started = time.perf_counter()
    models, datasets = models or CONFIG["models"], datasets or CONFIG["datasets"]
    cells = [(m, d) for m in models for d in datasets]
    fit_jobs = sorted({(m, d, a["fit_as"]) for m, d in cells for a in CONFIG["arms"].values()})
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        fitted = dict(pool.imap_unordered(fit_task, fit_jobs))
        jobs = [(m, d, arm, s, fitted[(m, d, a["fit_as"])]) for m, d in cells for arm, a in CONFIG["arms"].items() for s in SPLITS]
        # Longest first: VOI arms on the largest test splits.
        jobs.sort(key=lambda j: (Spec.of(j[2]).stop != "voi", j[3] != CONFIG["eval_split"], j[1] == "math"))
        traces = dict(pool.imap_unordered(trace_task, jobs))
    for m, d in cells:
        data = load(m, d)
        cell_traces = {(arm, s): traces[(m, d, arm, s)] for arm in CONFIG["arms"] for s in SPLITS}
        cell_fitted = {a: fitted[(m, d, a)] for (mm, dd, a) in fitted if (mm, dd) == (m, d)}
        result = assemble(m, d, data, cell_fitted, cell_traces, time.perf_counter() - started)
        write_json(result_path(m, d), result)
        path = traces_path(m, d)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"{arm}.{s}.{k}": v for (arm, s), tr in cell_traces.items() for k, v in tr.items()})
        log("v2-cell", model=m, dataset=d, primary=result["selected"]["accuracy"][CONFIG["primary"]]["test"])


def load_traces(model, dataset, arms=None, splits=SPLITS) -> dict:
    """{(arm, split): {chosen, confidence, other[, voi]}} from artifacts/v2/traces/<cell>.npz."""
    out = {}
    with np.load(traces_path(model, dataset)) as f:
        for key in f.files:
            arm, split, name = key.split(".")
            if (arms is None or arm in arms) and split in splits:
                out.setdefault((arm, split), {})[name] = f[key]
    return out


def rescore(model, dataset) -> dict:
    """Re-run the evaluation protocol from saved traces and fitted parameters (no refitting, no VOI simulation)."""
    from pgm_asc.utils import read_json
    previous = read_json(result_path(model, dataset))
    result = assemble(model, dataset, load(model, dataset), previous["fitted"], load_traces(model, dataset), previous["seconds"])
    write_json(result_path(model, dataset), result)
    return result
