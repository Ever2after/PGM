"""Figures: PGM mixture vs. baselines (paired deltas), accuracy vs. output tokens, and every ablation."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator  # noqa: E402

from pgm_asc.config import RESULTS  # noqa: E402

MODELS = (("gemma4b", "Gemma-3-4B-IT"), ("qwen3b", "Qwen2.5-3B-Instruct"), ("qwen7b", "Qwen2.5-7B-Instruct"),
          ("llama8b", "Llama-3.1-8B-Instruct"))
# (file stem suffix, heading, evaluations); in-domain and cross-dataset transfer are plotted separately.
SECTIONS = (("", "in-domain", (("math", "MATH500"), ("gsm8k", "GSM8K"), ("omnimath", "Omni-MATH"))),
            ("_transfer", "MATH → Omni-MATH transfer", (("math_to_omnimath", "Omni-MATH (MATH policy)"),)))
PGM = "mixture_s42__selected"

# Validated categorical slots 1-3 (all-pairs, light surface); shapes are the secondary encoding.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BASELINES = (("sc16", "SC16", "#2a78d6", "o"),
             ("reasc_offline", "ReASC offline", "#eb6834", "s"),
             ("reasc_dev_tuned", "ReASC dev-tuned", "#1baf7a", "D"))
OTHER = (("asc", "ASC", "^"), ("esc", "ESC", "v"), ("reasc_online", "ReASC online", "<"),
         ("asc_dev_tuned", "ASC dev-tuned", ">"))
# Ablation figure: color = method family (slots 1-2), shape = arm within the family, fill = operating point.
STRUCTURE = ("#2a78d6", (("uniform", "Uniform mixture", "o"), ("random", "Matched random", "s"), ("alpha1", "Alpha = 1", "D"),
                         ("single_partition", "Single partition", "^"), ("all_one", "All-one", "v"),
                         ("singletons", "Singletons", "<"), ("same_head_single", "Same-head single", ">")))
CONTROLS = ("#eb6834", (("candidate_counts", "Count MLP", "o"), ("candidate_confidence", "Confidence MLP", "s"),
                        ("deepsets_confidence", "DeepSets confidence", "D"), ("deepsets_embedding", "DeepSets embedding", "^")))
REFERENCES = (("sc16", "SC16", "o"), ("reasc_offline", "ReASC offline", "s"),
              ("reasc_dev_tuned", "ReASC dev-tuned", "D"), ("asc_dev_tuned", "ASC dev-tuned", ">"))
FOOT = ("Primary seed 42; PGM mixture at its dev-selected threshold. Output tokens include both answer-extraction passes "
        "and exclude input processing and the teacher-forcing feature pass.\nIn-domain Omni-MATH heads are fitted on "
        "Omni-MATH; transfer panels apply each model's saved MATH policy to the same Omni test responses.")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def deltas(metrics, evals):
    """Rows of (model, label, y, {baseline: contrast}) top to bottom, with a gap between models."""
    rows, y = [], 0.0
    for model, display in MODELS:
        for task, label in evals:
            key = f"{model}_{task}"
            if key in metrics:
                rows.append((display, label, y, metrics[key]["baseline_contrasts"]))
                y += 1
        y += 0.7
    return rows


def ratio_label(value, _):
    return f"{value:g}×"


def delta_figure(metrics, out, evals, suffix, heading):
    rows = deltas(metrics, evals)
    fig, (acc, tok) = plt.subplots(1, 2, figsize=(11, 2.2 + 0.5 * len(rows)), sharey=True, gridspec_kw={"wspace": 0.08})
    offsets = (-0.24, 0.0, 0.24)
    for (name, label, color, marker), dy in zip(BASELINES, offsets):
        for _, _, y, contrasts in rows:
            c = contrasts[name]
            lo, hi = c["accuracy_difference_ci95"]
            acc.plot([lo * 100, hi * 100], [y + dy] * 2, color=color, linewidth=1.6, solid_capstyle="round")
            acc.plot(c["accuracy_difference"] * 100, y + dy, marker=marker, color=color, markersize=6.5,
                     markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="none")
            lo, hi = c["output_token_ratio_ci95"]
            tok.plot([lo, hi], [y + dy] * 2, color=color, linewidth=1.6, solid_capstyle="round")
            tok.plot(c["output_token_ratio"], y + dy, marker=marker, color=color, markersize=6.5,
                     markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="none")
    for ax, ref in ((acc, 0.0), (tok, 1.0)):
        style(ax)
        ax.axvline(ref, color=INK2, linewidth=1.0, zorder=1)
    acc.set_xlim(-4.5, 7.0)
    acc.set_xlabel("Accuracy difference, PGM − baseline (percentage points)")
    acc.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    tok.set_xscale("log")
    tok.set_xlim(0.1, 3.2)
    tok.xaxis.set_major_locator(FixedLocator([0.125, 0.25, 0.5, 1, 2]))
    tok.xaxis.set_minor_locator(NullLocator())
    tok.xaxis.set_major_formatter(FuncFormatter(ratio_label))
    tok.set_xlabel("Output tokens, PGM ÷ baseline (log scale)")
    for ax, text, x, ha in ((acc, "PGM more accurate →", 0.985, "right"), (acc, "← baseline more accurate", 0.015, "left"),
                            (tok, "← PGM uses fewer tokens", 0.015, "left"), (tok, "PGM uses more →", 0.985, "right")):
        ax.text(x, 1.008, text, transform=ax.transAxes, ha=ha, va="bottom", color=MUTED, fontsize=8.5)
    acc.set_yticks([y for *_, y, _ in rows], [label for _, label, _, _ in rows])
    acc.set_ylim(rows[-1][2] + 0.6, rows[0][2] - 1.0)
    seen = set()
    for display, _, y, _ in rows:
        if display not in seen:
            seen.add(display)
            acc.text(-0.215, y - 0.62, display, transform=acc.get_yaxis_transform(), fontweight="bold",
                     color=INK, fontsize=9.5, ha="left", va="bottom")
    handles = [Line2D([], [], color=color, marker=marker, markersize=6.5, markeredgecolor=SURFACE, linewidth=1.6,
                      label=f"vs {label}") for _, label, color, marker in BASELINES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.125, 0.925), ncol=3, frameon=False,
               fontsize=9, labelcolor=INK2, handlelength=2.2, columnspacing=1.8)
    fig.suptitle(f"PGM mixture vs. baselines, {heading}",
                 x=0.125, y=0.985, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.125, 0.945, "Dots: point estimate. Whiskers: question-paired bootstrap 95% CI (10,000 replicates).",
             ha="left", color=INK2, fontsize=9)
    fig.text(0.125, 0.015, FOOT, ha="left", va="bottom", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(left=0.125, right=0.98, top=0.865, bottom=0.115)
    save(fig, out / f"pgm_vs_baselines_deltas{suffix}")


def panels(evals, panel_w, panel_h, header_in, left_in=1.3, footer_in=0.9):
    """Models x evaluations grid; a single evaluation is laid out as one row of models.

    Header/footer bands are fixed in inches so titles and legends never collide with panels.
    """
    single = len(evals) == 1
    rows, cols = (1, len(MODELS)) if single else (len(MODELS), len(evals))
    width, height = left_in + 0.3 + cols * panel_w, header_in + footer_in + rows * panel_h
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(left=left_in / width, right=1 - 0.3 / width, top=1 - header_in / height,
                        bottom=footer_in / height, hspace=0.32, wspace=0.24)
    cells = []
    for m, (model, display) in enumerate(MODELS):
        for e, (task, label) in enumerate(evals):
            r, c = (0, m) if single else (m, e)
            title = display if single else (label if r == 0 else None)
            ylabel = "accuracy" if single and c == 0 else (f"{display}\naccuracy" if not single and c == 0 else None)
            cells.append((axes[r, c], f"{model}_{task}", title, ylabel, r == rows - 1))
    return fig, cells, (left_in / width, header_in, height)


def finish_panel(ax, title, ylabel, bottom):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}k"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontweight="bold", labelpad=8)
    if bottom:
        ax.set_xlabel("Mean output tokens per question")


def tradeoff_figure(metrics, out, evals, suffix, heading):
    fig, cells, (x0, header, height) = panels(evals, 3.6, 2.5, header_in=1.3)
    for ax, key, title, ylabel, bottom in cells:
        style(ax, grid_axis="both")
        s = metrics[key]["summary"]
        point = lambda name: (s[name]["mean_output_tokens"] / 1000, s[name]["correct"] * 100)
        for name, _, marker in OTHER:
            ax.plot(*point(name), marker=marker, markersize=6.5, markerfacecolor=SURFACE,
                    markeredgecolor=MUTED, markeredgewidth=1.1, linestyle="none")
        for name, _, color, marker in BASELINES:
            ax.plot(*point(name), marker=marker, color=color, markersize=7.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, linestyle="none")
        ax.plot(*point(PGM), marker="*", color=INK, markersize=13, markeredgecolor=SURFACE,
                markeredgewidth=1.0, linestyle="none", zorder=5)
        ax.margins(x=0.12, y=0.18)
        finish_panel(ax, title, ylabel, bottom)
    top = lambda inches: 1 - inches / height
    handles = [Line2D([], [], marker="*", color=INK, markersize=12, markeredgecolor=SURFACE, linestyle="none",
                      label="PGM mixture")]
    handles += [Line2D([], [], marker=m, color=c, markersize=7.5, markeredgecolor=SURFACE, linestyle="none", label=l)
                for _, l, c, m in BASELINES]
    handles += [Line2D([], [], marker=m, markerfacecolor=SURFACE, markeredgecolor=MUTED, color=MUTED, markersize=6.5,
                       linestyle="none", label=l) for _, l, m in OTHER]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0, top(0.75)), ncol=8,
               frameon=False, fontsize=9, labelcolor=INK2, handletextpad=0.3, columnspacing=1.3)
    fig.suptitle(f"Accuracy vs. output tokens ({heading}): up and to the left is better",
                 x=x0, y=top(0.15), ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(x0, top(0.5), "Each panel has its own axes. SC16 always uses 16 responses; adaptive methods stop early.",
             ha="left", va="top", color=INK2, fontsize=9)
    fig.text(x0, 0.15 / height, FOOT, ha="left", va="bottom", color=MUTED, fontsize=7.5)
    save(fig, out / f"accuracy_vs_tokens{suffix}")


def ablation_figure(metrics, out, evals, suffix, heading):
    fig, cells, (x0, header, height) = panels(evals, 4.2, 2.9, header_in=2.0 if len(evals) > 1 else 2.6)
    for ax, key, title, ylabel, bottom in cells:
        style(ax, grid_axis="both")
        s = metrics[key]["summary"]
        point = lambda name: (s[name]["mean_output_tokens"] / 1000, s[name]["correct"] * 100)
        for name, _, marker in REFERENCES:
            ax.plot(*point(name), marker=marker, color="#b5b3ac", markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=0.8, linestyle="none", zorder=2)
        for color, arms in (STRUCTURE, CONTROLS):
            for arm, _, marker in arms:
                for op, filled in (("selected", True), ("fixed95", False)):
                    ax.plot(*point(f"{arm}_s42__{op}"), marker=marker, markersize=6, linestyle="none", zorder=3,
                            markerfacecolor=color if filled else SURFACE, markeredgecolor=SURFACE if filled else color,
                            markeredgewidth=0.8 if filled else 1.3)
        sel, fix = point("mixture_s42__selected"), point("mixture_s42__fixed95")
        ax.plot([sel[0], fix[0]], [sel[1], fix[1]], color=INK, linewidth=0.9, alpha=0.5, zorder=4)
        ax.plot(*sel, marker="*", color=INK, markersize=13, markeredgecolor=SURFACE, markeredgewidth=1.0,
                linestyle="none", zorder=6)
        ax.plot(*fix, marker="*", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=12,
                markeredgewidth=1.3, linestyle="none", zorder=6)
        ax.margins(x=0.1, y=0.15)
        finish_panel(ax, title, ylabel, bottom)
    top = lambda inches: 1 - inches / height

    def entry(label, marker, color, filled=True, size=6.5):
        return Line2D([], [], marker=marker, linestyle="none", markersize=size, label=label,
                      markerfacecolor=color if filled else SURFACE, markeredgecolor=SURFACE if filled else color,
                      markeredgewidth=0.8 if filled else 1.3)

    legends = (
        ("PGM mixture & operating point", [entry("Mixture, dev-selected γ", "*", INK, size=12),
                                           entry("Mixture, fixed γ = 0.95", "*", INK, False, 11),
                                           entry("Ablations: filled = dev-selected", "o", MUTED),
                                           entry("hollow = fixed 0.95", "o", MUTED, False)]),
        ("Partition-structure ablations", [entry(l, m, STRUCTURE[0]) for _, l, m in STRUCTURE[1]]),
        ("Learned controls without partitions", [entry(l, m, CONTROLS[0]) for _, l, m in CONTROLS[1]]),
        ("Baselines (single operating point)", [entry(l, m, "#b5b3ac") for _, l, m in REFERENCES]),
    )
    # Legend rows: a bold family label, then that family's entries (wrapped when the figure is narrow).
    y, per_row = 0.8, len(legends[1][1]) if len(evals) > 1 else 4
    for title, handles in legends:
        fig.text(x0, top(y), title, ha="left", va="center", color=INK, fontsize=9, fontweight="bold")
        for start in range(0, len(handles), per_row):
            fig.legend(handles=handles[start:start + per_row], loc="center left", bbox_to_anchor=(x0 + 2.6 / fig.get_figwidth(), top(y)),
                       ncol=per_row, frameon=False, fontsize=9, labelcolor=INK2, handletextpad=0.3, columnspacing=1.4, borderaxespad=0)
            y += 0.27
    fig.suptitle(f"Accuracy vs. output tokens with every learned ablation ({heading}), dev-selected and fixed 0.95",
                 x=x0, y=top(0.12), ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(x0, top(0.45), "Up and to the left is better. Each panel has its own axes; the line joins mixture's two operating points.",
             ha="left", va="top", color=INK2, fontsize=9)
    fig.text(x0, 0.15 / height, FOOT.replace("PGM mixture at its dev-selected threshold", "learned arms at both thresholds"),
             ha="left", va="bottom", color=MUTED, fontsize=7.5)
    save(fig, out / f"accuracy_vs_tokens_ablations{suffix}")


def save(fig, stem):
    for suffix in (".png", ".svg"):
        fig.savefig(stem.with_suffix(suffix), dpi=200)
    plt.close(fig)
    print(f"Wrote {stem}.png/.svg")


def load_evaluations():
    return {p.stem: json.loads(p.read_text()) for p in sorted((RESULTS / "evaluations").glob("*.json"))}


def make_figures(output=RESULTS / "figures"):
    metrics = load_evaluations()
    output.mkdir(parents=True, exist_ok=True)
    for suffix, heading, evals in SECTIONS:
        delta_figure(metrics, output, evals, suffix, heading)
        tradeoff_figure(metrics, output, evals, suffix, heading)
        ablation_figure(metrics, output, evals, suffix, heading)
