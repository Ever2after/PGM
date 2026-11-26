"""results/v2: README.md (tables), summary.csv (every selected operating point), figures (accuracy vs tokens)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pgm_asc.config import RESULTS
from pgm_asc.utils import read_json
from pgm_asc.v2 import CONFIG
from pgm_asc.v2.model import readable

NAMES = {"sc": "SC@k", "asc": "ASC", "esc": "ESC", "reasc_offline": "ReASC-off", "reasc_online": "ReASC-on",
         "M0": "M0 (γ stop)", "M1": "M1 PGM (VOI)", "M2": "M2 +confidence", "M3": "M3 +conf +copy",
         "M1_map": "MAP answer", "M1_no_other": "−OTHER", "M1_theta_const": "θ constant", "M1_no_crp": "−CRP (flat wrong)",
         "M2_map": "M2, MAP answer", "M1_copy": "+copy (text)", "M1_copy_hidden": "+copy (hidden)", "M1_copy_perm": "+copy (permuted)"}
DATASET = {"math": "MATH500", "gsm8k": "GSM8K", "omnimath": "Omni-MATH"}
MODEL = {"gemma4b": "Gemma-3-4B", "qwen7b": "Qwen2.5-7B", "llama8b": "Llama-3.1-8B"}
# Reference categorical palette (fixed order) + neutral gray for the SC@k reference curve.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
BASELINES = ["asc", "esc", "reasc_offline", "reasc_online"]


def cells():
    found = [read_json(p) for p in sorted((RESULTS / "v2" / "cells").glob("*.json"))]
    order = {(m, d): i for i, (m, d) in enumerate((m, d) for m in CONFIG["models"] for d in CONFIG["datasets"])}
    return sorted(found, key=lambda r: order.get((r["model"], r["dataset"]), len(order)))


def fmt_point(sel, sc):
    t = sel["test"]
    return f"{100 * t['accuracy']:.1f} / {100 * t['mean_tokens'] / sc['mean_tokens']:.0f}%"


def table_selected(rs, rule, methods):
    head = "| cell | SC16 acc | " + " | ".join(NAMES[m] for m in methods) + " |"
    lines = [head, "|" + "---|" * (len(methods) + 2)]
    for r in rs:
        sc = r["sc16"]["test"]
        row = [f"{MODEL.get(r['model'], r['model'])} {DATASET[r['dataset']]}", f"{100 * sc['accuracy']:.1f}"]
        row += [fmt_point(r["selected"][rule][m], sc) for m in methods]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def table_defaults(rs):
    methods = list(rs[0]["selected"]["default"])
    lines = ["| cell | " + " | ".join(f"{NAMES[m]}@{rs[0]['selected']['default'][m]['threshold']}" for m in methods) + " |",
             "|" + "---|" * (len(methods) + 1)]
    for r in rs:
        sc = r["sc16"]["test"]
        lines.append(f"| {MODEL.get(r['model'], r['model'])} {DATASET[r['dataset']]} | " +
                     " | ".join(fmt_point(r["selected"]["default"][m], sc) for m in methods) + " |")
    return "\n".join(lines)


def table_contrasts(rs, rule, against):
    lines = ["| cell | vs | Δacc (pp) [95% CI] | token ratio [95% CI] | McNemar p |", "|---|---|---|---|---|"]
    for r in rs:
        for m in against:
            c = r["contrasts"][rule].get(m)
            if c is None:
                continue
            lo, hi = c["accuracy_difference_ci95"]
            tl, th = c["token_ratio_ci95"]
            name = NAMES.get(m.split("@")[0], m) + ("@default" if "@" in m else "")
            lines.append(f"| {MODEL.get(r['model'], r['model'])} {DATASET[r['dataset']]} | {name} | {100 * c['accuracy_difference']:+.1f} "
                         f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {c['token_ratio']:.2f} [{tl:.2f}, {th:.2f}] | {c['mcnemar_exact_p']:.3f} |")
    return "\n".join(lines)


def table_frontier(rs, methods):
    lines = ["| cell | " + " | ".join(NAMES[m] for m in methods) + " |", "|" + "---|" * (len(methods) + 1)]
    for r in rs:
        f = r["frontier_score"]
        best = max(f[m] for m in methods)
        lines.append(f"| {MODEL.get(r['model'], r['model'])} {DATASET[r['dataset']]} | " +
                     " | ".join((f"**{100 * f[m]:.1f}**" if f[m] == best else f"{100 * f[m]:.1f}") for m in methods) + " |")
    mean = {m: np.mean([r["frontier_score"][m] for r in rs]) for m in methods}
    lines.append("| mean | " + " | ".join(f"{100 * mean[m]:.1f}" for m in methods) + " |")
    return "\n".join(lines)


def table_fits(rs):
    lines = ["| cell | arm | params | train NLL/question | fitted values |", "|---|---|---|---|---|"]
    for r in rs:
        for arm, f in r["fitted"].items():
            if arm not in ("M0", "M2", "M1_copy", "M3"):
                continue
            v = readable(f["params"])
            show = {"a": v["a"], "b": v["b"], "θ(q=.1)": v["theta(q=0.1)"], "θ(q=.9)": v["theta(q=0.9)"]}
            for k in ("b1", "b2", "s"):
                if k in v:
                    show[k] = v[k][0]
            if "rho" in v:
                show["ρ"], show["κ"] = v["rho"], v["kappa"]
            lines.append(f"| {MODEL.get(r['model'], r['model'])} {DATASET[r['dataset']]} | {arm} | {f['fit']['parameters']} | "
                         f"{f['fit']['nll_per_question']:.2f} | " + ", ".join(f"{k}={v:.2f}" for k, v in show.items()) + " |")
    return "\n".join(lines)


def figure(rs, series, path, title):
    if len(series) > len(SLOTS):
        raise ValueError(f"Draw at most {len(SLOTS)} methods per figure; split the method list")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "serif", "mathtext.fontset": "stix"})
    models = list(dict.fromkeys(r["model"] for r in rs))
    datasets = list(dict.fromkeys(r["dataset"] for r in rs))
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(3.4 * len(models), 2.8 * len(datasets)), squeeze=False)
    for r in rs:
        ax = axes[datasets.index(r["dataset"])][models.index(r["model"])]
        sc = r["sc16"]["test"]
        pts = sorted(r["curves"]["test"]["sc"], key=lambda p: p["mean_tokens"])
        ax.plot([p["mean_tokens"] / sc["mean_tokens"] for p in pts], [100 * p["accuracy"] for p in pts], color=GRAY, lw=1.2,
                marker="o", ms=3, label="SC@k", zorder=1)
        for i, m in enumerate(series):
            pts = sorted(r["curves"]["test"][m], key=lambda p: p["mean_tokens"])
            ax.plot([p["mean_tokens"] / sc["mean_tokens"] for p in pts], [100 * p["accuracy"] for p in pts], color=SLOTS[i],
                    lw=2 if i == 0 else 1.3, label=NAMES[m], zorder=3 if i == 0 else 2)
            sel = r["selected"]["cf_accuracy"][m]["test"]
            ax.plot(sel["mean_tokens"] / sc["mean_tokens"], 100 * sel["accuracy"], marker="D", ms=6, color=SLOTS[i],
                    markeredgecolor="white", markeredgewidth=1, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(left=max(.04, min(p["mean_tokens"] for p in r["curves"]["test"]["sc"]) / sc["mean_tokens"] * .8), right=1.15)
        lo = 100 * min(p["accuracy"] for p in r["curves"]["test"]["sc"])
        ax.set_ylim(lo - 1, 100 * max(max(p["accuracy"] for p in r["curves"]["test"][m]) for m in series + ["sc"]) + 1)
        ax.grid(True, color="#e6e5e0", lw=.6)
        ax.set_title(f"{MODEL.get(r['model'], r['model'])} · {DATASET[r['dataset']]}", fontsize=9, color=INK)
        ax.set_xticks([.1, .2, .3, .5, 1])
        ax.set_xticklabels(["10%", "20%", "30%", "50%", "100%"])
        ax.set_xlabel("output tokens (share of SC16)")
        ax.set_ylabel("test accuracy (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_columns = min(4, len(labels))
    fig.legend(handles, labels, loc="upper center", ncol=legend_columns, frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.suptitle(title + "  (◆ = operating point by 2-fold cross-fitting: fewest tokens with accuracy ≥ SC16 − 1 pp)",
                 y=1.035, fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .90 if len(labels) > legend_columns else .96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def load_all(methods=None, cells_filter=None):
    """Cell results with evaluated user methods merged in; returns (cells, names of the added methods)."""
    from pgm_asc.v2.methods import merge_into
    rs = cells()
    if cells_filter:
        rs = [r for r in rs if f"{r['model']}_{r['dataset']}" in cells_filter]
    if not rs:
        raise SystemExit("No v2 cell results; run `pgm v2 run` (or download results) first")
    return rs, merge_into(rs, NAMES, methods)


def plot(methods, out, cells_filter=None, title="accuracy vs output tokens (test)"):
    """One accuracy-token figure (PNG + SVG) with any methods: PGM arms, baselines or evaluated user methods."""
    rs, _ = load_all(None, cells_filter)
    unknown = [m for m in methods if m not in rs[0]["curves"]["test"]]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; available: {sorted(rs[0]['curves']['test'])}")
    figure(rs, list(methods), Path(out), title)
    return Path(out)


def overview_groups(extra):
    """Keep the published palette while allowing any number of plugin methods."""
    base=["M1", "asc", "esc", "reasc_offline"]
    width=len(SLOTS)-len(base)
    return [base+extra[i:i+width] for i in range(0,len(extra),width)] or [base]


def report(methods=None):
    rs, extra = load_all(methods)
    out = RESULTS / "v2"
    primary, ladder = CONFIG["primary"], CONFIG["ladder"]
    ablations = [a for a in CONFIG["arms"] if a not in ladder]
    main = [primary] + [m for m in ladder if m != primary]
    base = ["asc", "esc", "reasc_offline", "reasc_online"] + extra
    overview=[]
    for i,group in enumerate(overview_groups(extra)):
        stem='accuracy_vs_tokens'+(f'_{i+1}' if i else '')
        figure(rs, group, out / "figures" / stem, "PGM v2 (M1) vs baselines")
        overview.append(f"![accuracy vs tokens](figures/{stem}.png)")
    figure(rs, main, out / "figures" / "ladder", "PGM v2 ladder")
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "dataset", "rule", "method", "threshold", "dev_accuracy", "dev_mean_tokens", "test_accuracy",
                    "test_mean_tokens", "test_mean_samples", "test_token_share_of_sc16", "frontier_score"])
        for r in rs:
            for rule, methods in r["selected"].items():
                for m, x in methods.items():
                    d = x.get("dev", {})
                    w.writerow([r["model"], r["dataset"], rule, m, x["threshold"], d.get("accuracy", ""), d.get("mean_tokens", ""),
                                x["test"]["accuracy"], x["test"]["mean_tokens"], x["test"]["mean_samples"],
                                x["test"]["mean_tokens"] / r["sc16"]["test"]["mean_tokens"], r["frontier_score"][m]])
    budgets = [f"budget_{b}" for b in CONFIG["selection"]["budgets"]]
    defaults = [f"{m}@default" for m in ["sc", "asc", "esc", "reasc_offline", "reasc_online"] + extra]
    text = [
        "# PGM v2 results", "",
        "Generated by `pgm v2 report` from `results/v2/cells/*.json`. Table cells: test accuracy (%) / output tokens as a share of SC16.",
        f"Primary model: **{NAMES[primary]}**. Method and protocol: `docs/METHOD.md`.", "",
        "Operating points are selected two ways, never using a question's own label: on **dev** (200 questions; easier than test "
        "on these benchmarks), and by **2-fold cross-fitting on test** (threshold chosen on one half, applied to the other).", "",
        *overview, "",
        "## 1. Frontier score", "",
        "Mean, over 60 budgets from SC@2 to SC16 tokens, of the best test accuracy within the budget (oracle summary, identical for every method).", "",
        table_frontier(rs, main + base + ["sc"]), "",
        "## 2. Matched accuracy: fewest tokens with accuracy ≥ SC16 − 1 pp", "",
        "### Cross-fitted on test", "", table_selected(rs, "cf_accuracy", main + base), "",
        "### Selected on dev", "", table_selected(rs, "accuracy", main + base), "",
        "### Published defaults (no tuning)", "", table_defaults(rs), "",
        f"## 3. {NAMES[primary]} against baselines (paired, question level; positive Δ = PGM better, token ratio < 1 = PGM cheaper)", "",
        "### Cross-fitted matched accuracy", "", table_contrasts(rs, "cf_accuracy", base + defaults), "",
        "### Dev-selected matched accuracy", "", table_contrasts(rs, "accuracy", base + defaults), "",
        "## 4. Is each part of the graph needed?", "",
        "![ladder](figures/ladder.png)", "",
        "Frontier score of the primary model and each variant:", "",
        table_frontier(rs, [primary] + [m for m in ladder if m != primary] + ablations), "",
        "Paired contrasts at the cross-fitted matched-accuracy point:", "",
        table_contrasts(rs, "cf_accuracy", [m for m in ladder if m != primary] + ablations), "",
        "## 5. Matched budget: best accuracy within a share of SC16 tokens (cross-fitted)", "",
    ]
    for b in budgets:
        text += [f"### {b.split('_')[1]} × SC16 tokens", "", table_selected(rs, f"cf_{b}", main + ["asc", "esc", "reasc_offline"] + extra), ""]
    text += ["## 6. Fitted parameters", "", table_fits(rs), "",
             "## 7. Calibration of the reported posterior (all test prefixes, 10-bin ECE)", "",
             "| cell | " + " | ".join(NAMES[a] for a in main) + " |", "|" + "---|" * (len(main) + 1)]
    for r in rs:
        text.append(f"| {MODEL.get(r['model'], r['model'])} {DATASET[r['dataset']]} | " +
                    " | ".join(f"{r['calibration'][a]['ece_10bin_all_prefixes']:.3f}" for a in main) + " |")
    (out / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    from pgm_asc.v2.html import make_report
    make_report(out / "report.html", rs, extra)
    print(f"Wrote {out}/README.md, summary.csv, figures/, report.html")
