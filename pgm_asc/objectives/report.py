"""Readable objective-ablation report and complete machine-readable tables."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from pgm_asc import config
from pgm_asc.diagnostics.report import csv_file, fmt, markdown_html, table

ORDER = ("released_M1", "joint", "prefix_uniform", "mixed_uniform", "prefix_v1", "mixed_v1")
LABELS = {"released_M1": "Released M1 (reference)", "joint": "Joint",
          "prefix_uniform": "Prefix / uniform", "mixed_uniform": "Mixed / uniform",
          "prefix_v1": "Prefix / v1 weights", "mixed_v1": "Mixed / v1 weights"}
COLORS = {"joint": "#3d6fa4", "prefix_uniform": "#d28136", "mixed_uniform": "#34846c"}


def make_figures(out, cells, colors=None, labels=None):
    colors = COLORS if colors is None else colors
    labels = LABELS if labels is None else labels
    cache = config.ARTIFACTS / "objective_ablation" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    datasets = sorted({c["dataset"] for c in cells})
    split = "test" if "test" in cells[0]["splits"] else next(iter(cells[0]["splits"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.7), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for name in colors:
            rr = [r for c in cells if c["dataset"] == ds for r in c["splits"][split]["models"][name]["posterior_by_prefix"]]
            ts = sorted({r["prefix"] for r in rr})
            ax.plot(ts, [np.mean([r["unique_nll"] for r in rr if r["prefix"] == t]) for t in ts],
                    marker="o", color=colors[name], label=labels[name])
        ax.set(title=ds, xlabel="Observed responses", ylabel="Unique-correct NLL (lower is better)")
        ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "posterior.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 7), squeeze=False)
    for col, ds in enumerate(datasets):
        for row, event in enumerate(("repair", "damage")):
            ax = axes[row, col]
            for name in colors:
                rr = [r for c in cells if c["dataset"] == ds for r in c["splits"][split]["models"][name]["transition_calibration"] if r["event"] == event]
                points = []
                for lower in sorted({r["lower"] for r in rr}):
                    bins = [r for r in rr if r["lower"] == lower]
                    n = sum(r["n"] for r in bins)
                    points.append([sum(r["prediction"] * r["n"] for r in bins) / n,
                                   sum(r["observed"] * r["n"] for r in bins) / n, n])
                if points:
                    x, y, n = np.array(points).T
                    ax.plot(x, y, color=colors[name], label=labels[name])
                    ax.scatter(x, y, s=10 + 70*n/n.max(), color=colors[name], alpha=.8)
            lo, hi = ax.get_xlim()
            ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=.6)
            ax.set(title=f"{ds}: {event}", xlabel="Predicted probability", ylabel="Observed frequency")
            ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "transitions.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 3.7), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for name in colors:
            rr = [r for c in cells if c["dataset"] == ds for r in c["splits"][split]["models"][name]["gain_calibration"]]
            points = []
            for lower in sorted({r["lower"] for r in rr}):
                bins = [r for r in rr if r["lower"] == lower]
                n = sum(r["n"] for r in bins)
                points.append([sum(r["prediction"]*r["n"] for r in bins)/n,
                               sum(r["observed"]*r["n"] for r in bins)/n, n])
            if points:
                x, y, n = np.array(points).T
                ax.plot(x, y, label=labels[name], color=colors[name])
                ax.scatter(x, y, s=10+70*n/n.max(), color=colors[name], alpha=.8)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=.6)
        ax.set(title=ds, xlabel="Predicted gain", ylabel="Observed gain")
        ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "gain.png", dpi=160)
    plt.close(fig)


def report(out):
    out = Path(out)
    manifest = json.loads((out / "run.json").read_text())
    cells = [json.loads((out / p).read_text()) for p in manifest["cells"]]
    if not cells or len({c["experiment_id"] for c in cells}) != 1:
        raise ValueError("Report requires a nonempty, single-protocol run")
    tables = {name: [] for name in ("summary", "posterior_by_prefix", "future_by_horizon", "posterior_calibration",
              "gain_calibration", "transition_calibration", "paired_contrasts", "mc_seed_sensitivity",
              "training_restarts", "selection", "parameters", "future_unique_summary")}
    for c in cells:
        base = {"model": c["model"], "dataset": c["dataset"]}
        fitted = json.loads((out / "selection" / f"{c['model']}_{c['dataset']}.json").read_text())
        for name, f in fitted["fits"].items():
            for restart in f["restarts"]:
                tables["training_restarts"].append({**base, "objective": name, **f["objective"],
                    **{k: v for k, v in restart.items() if k not in ("params", "initial")},
                    "chosen_restart": restart["restart_seed"] == f["best"]["restart_seed"]})
        for name, source in fitted["selected"].items():
            f = fitted["fits"][source]
            tables["selection"].append({**base, "method": name, "source": source, "lambda": f["objective"]["mixing"],
                "dev_unique_nll": f["best"]["dev_unique_nll"], "train_kept": fitted["train_kept"],
                "dev_unique": fitted["dev_unique"], "restart_seed": f["best"]["restart_seed"],
                "locked_at": fitted["selection_locked_at"]})
        for name, p in c["parameters"].items():
            tables["parameters"].append({**base, "method": name, "count": p["count"], **p["raw"], **p["readable"]})
        for split, s in c["splits"].items():
            for name, entry in s["models"].items():
                key = {**base, "split": split, "method": name}
                for field in ("summary", "future_unique_summary"):
                    for metric, value in entry[field].items():
                        ci = value["ci95"] or [None, None]
                        tables[field].append({**key, "metric": metric, "mean": value["mean"],
                                              "ci_low": ci[0], "ci_high": ci[1], "questions": value["questions"]})
                for field in ("posterior_by_prefix", "future_by_horizon", "posterior_calibration", "gain_calibration", "transition_calibration", "mc_seed_sensitivity"):
                    tables[field] += [{**key, **r} for r in entry[field]]
            for name, differences in s["paired_model_minus_joint"].items():
                for metric, value in differences.items():
                    ci = value["ci95"] or [None, None]
                    tables["paired_contrasts"].append({**base, "split": split, "method": name,
                        "comparison": "method minus joint", "metric": metric, "mean": value["mean"],
                        "ci_low": ci[0], "ci_high": ci[1], "questions": value["questions"]})
    for name, rows in tables.items():
        csv_file(out / f"{name}.csv", rows)
    macro = []
    for split in sorted({r["split"] for r in tables["summary"]}):
        for method in ORDER:
            for metric in sorted({r["metric"] for r in tables["summary"]}):
                values = [r["mean"] for r in tables["summary"] if r["split"] == split and r["method"] == method and r["metric"] == metric and r["mean"] is not None]
                if values:
                    macro.append({"split": split, "method": method, "metric": metric,
                                  "mean": float(np.mean(values)), "cells": len(values)})
            bias = [abs(r["mean"]) for r in tables["summary"] if r["split"] == split and r["method"] == method and r["metric"] == "gain_bias"]
            if bias:
                macro.append({"split": split, "method": method, "metric": "abs_cell_gain_bias",
                              "mean": float(np.mean(bias)), "cells": len(bias)})
    csv_file(out / "macro_summary.csv", macro)
    dataset_rows = []
    for split in sorted({r["split"] for r in tables["summary"]}):
        for ds in sorted({c["dataset"] for c in cells}):
            for method in ORDER:
                for metric in sorted({r["metric"] for r in tables["summary"]}):
                    vv = [r["mean"] for r in tables["summary"] if r["split"] == split and r["dataset"] == ds and r["method"] == method and r["metric"] == metric and r["mean"] is not None]
                    if vv:
                        dataset_rows.append({"split": split, "dataset": ds, "method": method,
                                             "metric": metric, "mean": float(np.mean(vv)), "cells": len(vv)})
    csv_file(out / "dataset_summary.csv", dataset_rows)
    primary_split = "test" if any(r["split"] == "test" for r in macro) else macro[0]["split"]
    values = {(r["method"], r["metric"]): r["mean"] for r in macro if r["split"] == primary_split}
    methods = [name for name in ORDER if (name, "unique_nll") in values]

    def score(name, metric, multiplier=1.):
        v = values.get((name, metric))
        return fmt(v * multiplier if v is not None else None)

    post_rows = [[LABELS[name]] + [score(name, k) for k in ("unique_nll", "reference_brier", "other_brier", "next_answer_nll")] for name in methods]
    future_rows = [[LABELS[name]] + [score(name, k) for k in ("repair_brier", "damage_brier", "gain_mse", "gain_mse_minus_zero")] + [score(name, "gain_bias", 100), score(name, "abs_cell_gain_bias", 100)] for name in methods]
    transitions = []
    for name in methods:
        entries = [c["splits"][primary_split]["models"][name] for c in cells]
        row = [LABELS[name]]
        for event in ("repair", "damage"):
            for key in ("prediction", "observed"):
                row.append(fmt(100*np.mean([np.mean([r[key] for r in e["future_by_horizon"] if r["event"] == event]) for e in entries]), 2))
        transitions.append(row)
    selection_rows = []
    for c in cells:
        rr = [r for r in tables["selection"] if r["model"] == c["model"] and r["dataset"] == c["dataset"]]
        by_name = {r["method"]: r for r in rr}
        selection_rows.append([f"{c['model']} / {c['dataset']}", c["train_kept"],
            by_name.get("mixed_uniform", {}).get("lambda", "N/A"), by_name.get("mixed_v1", {}).get("lambda", "N/A")])
    cell_rows = []
    for c in cells:
        m = c["splits"][primary_split]["models"]
        row = [f"{c['model']} / {c['dataset']}"]
        for name in ("joint", "prefix_uniform", "mixed_uniform"):
            row.append(fmt(m[name]["summary"]["unique_nll"]["mean"]))
        for name in ("joint", "prefix_uniform", "mixed_uniform"):
            row.append(fmt(100*m[name]["summary"]["gain_bias"]["mean"], 2))
        cell_rows.append(row)
    training = cells[0]["study"]["training"]
    evaluation = cells[0]["study"]["evaluation"]
    main = [name for name in ("joint", "prefix_uniform", "mixed_uniform") if name in methods]
    findings = []
    for name in main[1:]:
        findings.append(f"{LABELS[name]}: Joint 대비 Unique NLL 변화 {values[name,'unique_nll']-values['joint','unique_nll']:+.4f}, "
                        f"next-answer NLL 변화 {values[name,'next_answer_nll']-values['joint','next_answer_nll']:+.4f}, "
                        f"gain MSE 변화 {values[name,'gain_mse']-values['joint','gain_mse']:+.5f}. 각 loss는 음수 변화가 개선입니다.")
    text = ["# M1 학습 목적 ablation", "", "M1의 구조·입력·4개 parameter와 수치 grid를 고정하고 학습 목적만 비교했습니다. 기존 연구의 모델과 결과는 덮어쓰지 않았습니다.", "",
        f"범위: {len(cells)}개 cell. 주 결과는 {primary_split}, prefix={evaluation['prefixes']}, horizon={evaluation['horizons']}입니다.",
        f"미래 예측은 MC seed {len(evaluation['mc_seeds'])}개 × {evaluation['simulations_per_seed']} draws, 총 {len(evaluation['mc_seeds'])*evaluation['simulations_per_seed']}개 continuation을 사용합니다. 모든 방법에 같은 seed 규칙을 적용했습니다.", "",
        "## 고정한 실험 조건", "",
        "공통 train: 800문항 중 bank 전체에 정답 판정 후보가 최대 하나인 질문만 유지합니다. Joint/Prefix/Mixed 모두 같은 질문과 같은 latent 정답 target을 사용합니다. Dev 선택도 unique-correct 부분집합을 사용하며 test는 선택에 사용하지 않았습니다.",
        "Joint는 -log p(D16,c*)/16, Prefix는 16개 prefix의 가중 평균 posterior NLL입니다. Mixed=(Joint+λ×Prefix)/(1+λ)입니다. 정규화된 loss에 동일한 ridge=1e-4/16을 더합니다. 따라서 정규화된 Joint는 원래 M1의 ridge 강도와 수학적으로 같습니다.",
        f"혼합 λ 후보는 {training['lambda_grid']}입니다. 각 cell·weighting에서 dev의 고정 평가 prefix 평균 Unique NLL이 가장 낮은 λ를 선택했습니다. 미래 성능을 이용한 λ 선택은 하지 않았고, λ별 dev 점수는 selection/*.json과 training_restarts.csv에 남깁니다.",
        f"같은 초기값 {training['restart_seeds']}에서 시작해 각 목적의 penalized train loss가 가장 낮은 해를 채택했습니다. Full-batch L-BFGS이므로 이들은 optimizer 재시작이며 독립적인 학습 데이터 seed가 아닙니다.",
        "Uniform과 v1 weighting을 별도 block으로 비교합니다. 실제 v1 코드의 band별 총 가중치는 0.4/0.2/0.2/0.2입니다. Calibration temperature는 추가하지 않았습니다.",
        "Released M1은 전체 train과 기존 target으로 학습한 참조 모델입니다. 순수한 loss 비교의 기준은 공통 train에서 다시 학습한 Joint입니다.", "",
        "## Posterior와 미래 답 예측", "", "아래는 cell을 동일하게 가중한 macro 평균입니다. 모든 loss는 낮을수록 좋습니다. 다음 답의 categorical 분포는 Monte Carlo가 아닌 정확한 적분으로 계산합니다.", "",
        table(["방법", "Unique NLL", "Reference Brier", "OTHER Brier", "Next-answer NLL"], post_rows), "",
        "## 회복·손상·기대 이득", "", "Gain은 같은 continuation의 미래 정오−현재 정오를 평균합니다. Gain=repair−damage가 표본 수준에서 성립합니다. Gain bias는 예측−관측, 단위는 pp입니다. Gain MSE−zero가 음수면 항상 0-gain을 예측하는 기준보다 좋습니다.", "",
        table(["방법", "Repair Brier", "Damage Brier", "Gain MSE", "Gain MSE−zero", "Gain bias pp", "평균 절대 cell bias pp"], future_rows), "",
        table(["방법", "회복 예측 %", "실제 회복 %", "손상 예측 %", "실제 손상 %"], transitions), "",
        "## 관찰한 변화", "", *[x for p in findings for x in (p, "")],
        "이 수치만으로 한 학습 목적이 모든 층에서 우월하다고 판단하지 않습니다. Posterior·다음 답·회복/손상·gain에서 생긴 trade-off를 함께 확인해야 합니다.", "",
        "Gain bias의 전체 평균이 작아도 cell 간 양·음 오차가 상쇄될 수 있어, 평균 |cell bias|도 함께 표시했습니다. 데이터셋별 전체 점수는 dataset_summary.csv에 있습니다.", "",
        "![Posterior by prefix](figures/posterior.png)", "",
        "## Cell별 결과와 dev 선택", "",
        table(["Cell", "Train 유지", "Mixed uniform λ", "Mixed v1 λ"], selection_rows), "",
        table(["Cell", "Joint NLL", "Prefix NLL", "Mixed NLL", "Joint gain bias pp", "Prefix bias pp", "Mixed bias pp"], cell_rows), "",
        "## Calibration 곡선", "", "각 데이터셋에서 모델·prefix·horizon의 같은 구간을 합쳤습니다. 점이 클수록 관측 상태가 많습니다. 이 상태들은 독립 표본이 아닙니다.", "",
        "![Repair and damage calibration](figures/transitions.png)", "",
        "![Gain calibration](figures/gain.png)", "",
        "## 범위와 한계", "",
        "1. 주 uncertainty는 질문 단위 paired bootstrap 95% CI입니다. 반복 prefix/horizon·MC seed를 독립 질문으로 세지 않습니다. Macro 표에는 pooled CI나 다중비교 유의성 주장을 붙이지 않습니다.",
        "2. 혼합 λ는 dev posterior NLL로 골랐습니다. 따라서 미래 예측에 최적인 λ라고 보장하지 않습니다. 선택과 test 평가가 같은 질문을 사용하지 않으며, selection lock은 test를 읽기 전에 저장했습니다.",
        "3. 최적화 초기값 3개는 모든 목적에 동일하지만, 선택한 best fit의 test CI가 초기값·학습 bank·judge의 불확실성을 모두 포함하는 것은 아닙니다. MC seed별 점수와 모든 restart를 별도 CSV로 제공합니다.",
        "4. 주 라벨은 기존 xVerify candidate 판정입니다. 다중 정답 문자열 문제는 주 Unique NLL과 별개로 전체 reference·future 점수에 포함됩니다. future_unique_summary.csv에는 같은 unique-correct subset의 미래 점수도 제공합니다.",
        "5. 회복·손상 Brier가 MC event의 주 비교입니다. MC NLL은 유한 draw에서 발생하는 0/1 확률 때문에 민감할 수 있습니다. NLL clipping=1e-12와 zero_probability_events를 함께 보고합니다. 정확한 next-answer NLL에는 이 MC 문제가 없습니다.",
        "6. M1은 빈 답을 생성하는 observation model이 없어 해당 관측에 0 확률을 줍니다. 구조·입력을 유지하기 위해 이 한계도 유지했습니다. 실제 stopping criterion이나 토큰 절감 효과를 이번 실험에서 선택하지 않았습니다.",
        "7. 이미 탐색한 test bank의 비교이므로 독립적인 최종 검증이 아닙니다. 새 구조 또는 학습 목적을 확정한 뒤에는 대표성 있는 별도 질문 holdout이 필요합니다.", "",
        "## 재현", "", "pgm objective fit --workers 2", "pgm objective run --workers 2", "pgm objective report", "",
        "configs/objectives.json과 protocol.json에 설정을 저장합니다. selection/*.json은 test를 읽기 전의 λ·optimizer 선택 기록입니다. cells/*.json은 전체 점수와 provenance를 담고, 큰 posterior/forecast 배열은 artifacts/objective_ablation/ 아래에 저장합니다.",
        "summary.csv / macro_summary.csv / posterior_by_prefix.csv / future_by_horizon.csv / paired_contrasts.csv / gain_calibration.csv / transition_calibration.csv / training_restarts.csv / selection.csv / mc_seed_sensitivity.csv / parameters.csv를 제공합니다."]
    markdown = "\n".join(text) + "\n"
    (out / "README.md").write_text(markdown)
    make_figures(out, cells)
    style = "body{font-family:system-ui,sans-serif;max-width:1380px;margin:40px auto;padding:0 24px;line-height:1.7;background:#fafbfc;color:#17212b}h1,h2{line-height:1.3;margin-top:2em}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:9px;border-bottom:1px solid #dce2e8}th{background:#eaf0f7}img{width:100%;margin:18px 0}"
    (out / "report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>M1 학습 목적 ablation</title><style>"+style+"</style><body>"+markdown_html(markdown)+"</body></html>")
    return out / "report.html"
