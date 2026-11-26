"""Tables, figures and a standalone HTML report from diagnostic cell results."""
from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path

import numpy as np

LABELS = {"v1_mixture": "v1 mixture (calibrated)", "v1_mixture_raw": "v1 mixture (raw)", "v2_M1": "v2 M1"}


def csv_file(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def fmt(x, digits=4):
    return "N/A" if x is None else f"{x:.{digits}f}"


def summary_value(entry, metric):
    return entry.get("summary", {}).get(metric, {}).get("mean")


def figures(out, cells):
    from pgm_asc import config
    cache = config.ARTIFACTS / "diagnostics" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    datasets = sorted({c["dataset"] for c in cells})
    split = "test" if any("test" in c["splits"] for c in cells) else "dev"
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.6), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        relevant = [c for c in cells if c["dataset"] == dataset and split in c["splits"]]
        methods = sorted({name for c in relevant for name in c["splits"][split]["models"]})
        for name in methods:
            rows = [r for c in relevant for r in c["splits"][split]["models"].get(name, {}).get("posterior_by_prefix", [])]
            ts = sorted({r["prefix"] for r in rows})
            if ts:
                ax.plot(ts, [np.mean([r["reference_brier"] for r in rows if r["prefix"] == t]) for t in ts],
                        marker="o", label=LABELS.get(name, name))
        ax.set(title=dataset, xlabel="Observed responses", ylabel="Reference-answer Brier (lower is better)")
        ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "posterior_quality.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.6), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        relevant = [c for c in cells if c["dataset"] == dataset and split in c["splits"]]
        methods = sorted({name for c in relevant for name, e in c["splits"][split]["models"].items() if "gain_calibration" in e})
        for name in methods:
            rows = [r for c in relevant for r in c["splits"][split]["models"].get(name, {}).get("gain_calibration", [])]
            points = []
            for lower in sorted({r["lower"] for r in rows}):
                rr = [r for r in rows if r["lower"] == lower]
                n = sum(r["n"] for r in rr)
                points.append((sum(r["prediction"] * r["n"] for r in rr) / n,
                               sum(r["observed"] * r["n"] for r in rr) / n, n))
            if points:
                x, y, n = np.array(points).T
                line, = ax.plot(x, y, label=LABELS.get(name, name))
                ax.scatter(x, y, s=10 + 100 * n / n.max(), color=line.get_color(), alpha=.85)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=.6)
        ax.set(title=dataset, xlabel="Predicted accuracy gain", ylabel="Observed accuracy gain")
        ax.grid(alpha=.2)
    if axes[0, -1].get_legend_handles_labels()[0]:
        axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "gain_calibration.png", dpi=160)
    plt.close(fig)


def markdown_html(markdown):
    """Small renderer for this report's paragraphs, headings and simple tables."""
    lines, output, in_table = markdown.splitlines(), [], False
    for line in lines:
        if line.startswith("|"):
            parts = [html.escape(p.strip()) for p in line.strip("|").split("|")]
            if all(set(p) <= {"-", ":"} for p in parts):
                continue
            tag = "td" if in_table else "th"
            if not in_table:
                output.append("<div class='scroll'><table>")
                in_table = True
            output.append("<tr>" + "".join(f"<{tag}>{p}</{tag}>" for p in parts) + "</tr>")
            continue
        if in_table:
            output.append("</table></div>")
            in_table = False
        if line.startswith("!["):
            label, source = line[2:].split("](", 1)
            output.append(f'<img alt="{html.escape(label)}" src="{html.escape(source[:-1])}">')
        elif line.startswith("#"):
            n = len(line) - len(line.lstrip("#"))
            output.append(f"<h{n}>{html.escape(line[n:].strip())}</h{n}>")
        elif line:
            output.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        output.append("</table></div>")
    return "\n".join(output)


def report(out):
    run_path = out / "run.json"
    paths = json.loads(run_path.read_text())["cells"] if run_path.exists() else sorted((out / "cells").glob("*.json"))
    cells = [json.loads((Path(p) if Path(p).is_absolute() else out / p).read_text()) for p in paths]
    if not cells:
        raise ValueError("No diagnostic results to report")
    protocols = [c["protocol"] for c in cells]
    if any(p != protocols[0] for p in protocols[1:]):
        raise ValueError("Cannot aggregate cells evaluated with different protocols")
    summaries, posterior, future, gain_bins, posterior_bins, contrasts, capabilities = [], [], [], [], [], [], []
    for c in cells:
        for split, s in c["splits"].items():
            base = {"model": c["model"], "dataset": c["dataset"], "split": split}
            for name, entry in s["models"].items():
                key = {**base, "method": name}
                cap = entry["capabilities"]
                capabilities.append({**key, "posterior": cap.get("posterior", cap["status"]),
                                     "future": cap.get("future", "unavailable"),
                                     "reason": cap.get("future_reason", cap.get("reason", ""))})
                for metric, value in entry.get("summary", {}).items():
                    ci = value["ci95"] or [None, None]
                    summaries.append({**key, "metric": metric, "mean": value["mean"], "ci_low": ci[0], "ci_high": ci[1], "questions": value["questions"]})
                posterior += [{**key, **r} for r in entry.get("posterior_by_prefix", [])]
                future += [{**key, **r} for r in entry.get("future_by_horizon", [])]
                gain_bins += [{**key, **r} for r in entry.get("gain_calibration", [])]
                posterior_bins += [{**key, **r} for r in entry.get("posterior_calibration", [])]
            for name, entry in s["paired_M1_minus_model"].items():
                for metric, value in entry.items():
                    ci = value["ci95"] or [None, None]
                    contrasts.append({**base, "comparison": f"v2_M1 minus {name}", "metric": metric,
                                      "mean": value["mean"], "ci_low": ci[0], "ci_high": ci[1], "questions": value["questions"]})
    for name, rows in [("summary", summaries), ("posterior_by_prefix", posterior), ("future_by_horizon", future),
                       ("gain_calibration", gain_bins), ("posterior_calibration", posterior_bins),
                       ("paired_contrasts", contrasts), ("capabilities", capabilities)]:
        csv_file(out / f"{name}.csv", rows)
    macro = []
    for split in sorted({r["split"] for r in summaries}):
        for method in sorted({r["method"] for r in summaries}):
            for metric in sorted({r["metric"] for r in summaries}):
                rr = [r for r in summaries if r["split"] == split and r["method"] == method and r["metric"] == metric and r["mean"] is not None]
                if rr:
                    macro.append({"split": split, "method": method, "metric": metric,
                                  "mean": float(np.mean([r["mean"] for r in rr])), "cells": len(rr)})
    csv_file(out / "macro_summary.csv", macro)
    primary_split = "test" if any(r["split"] == "test" for r in summaries) else summaries[0]["split"]
    methods = sorted({r["method"] for r in summaries if r["split"] == primary_split})
    metrics_order = ["unique_nll", "reference_brier", "other_brier", "next_answer_nll", "gain_mse", "gain_mse_minus_zero"]
    macro_rows = []
    for method in methods:
        values = {r["metric"]: r["mean"] for r in macro if r["split"] == primary_split and r["method"] == method}
        macro_rows.append([LABELS.get(method, method)] + [fmt(values.get(k)) for k in metrics_order])
    comparison_rows, future_rows, audit_rows = [], [], []
    for c in cells:
        if primary_split not in c["splits"]:
            continue
        s = c["splits"][primary_split]
        a, b = s["models"].get("v1_mixture", {}), s["models"].get("v2_M1", {})
        diff = s["paired_M1_minus_model"].get("v1_mixture", {}).get("reference_brier", {})
        ci = diff.get("ci95")
        comparison_rows.append([f"{c['model']} / {c['dataset']}", fmt(summary_value(a, "reference_brier")),
                                fmt(summary_value(b, "reference_brier")), fmt(diff.get("mean")),
                                "N/A" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"])
        for name, entry in s["models"].items():
            if "future_by_horizon" not in entry:
                continue
            gains = [r for r in entry["future_by_horizon"] if r["event"] == "gain"]
            future_rows.append([f"{c['model']} / {c['dataset']}", name,
                                fmt(summary_value(entry, "next_answer_nll")),
                                fmt(float(np.mean([r["prediction"] for r in gains]))),
                                fmt(float(np.mean([r["observed"] for r in gains]))),
                                fmt(summary_value(entry, "gain_mse_minus_zero"))])
        audit = s["audit"]
        audit_rows.append([f"{c['model']} / {c['dataset']}", audit["questions"], audit["multiple_accepted_candidates"],
                           audit["no_correct_candidate_in_bank"], audit["invalid_responses"]])
    protocol = cells[0]["protocol"]
    findings = []
    common = [c["splits"][primary_split]["models"] for c in cells if primary_split in c["splits"]
              and "summary" in c["splits"][primary_split]["models"].get("v1_mixture", {})
              and "summary" in c["splits"][primary_split]["models"].get("v2_M1", {})]
    if common:
        wins = sum(summary_value(m["v1_mixture"], "reference_brier") < summary_value(m["v2_M1"], "reference_brier") for m in common)
        findings.append(f"공통 답의 정답 확률을 평가한 Reference Brier는 {len(common)}개 중 {wins}개 cell에서 calibrated v1이 더 낮습니다. 이는 현재 확률 품질의 결과이며 stopping 효용의 우열을 뜻하지 않습니다.")
    mv = {(r["method"], r["metric"]): r["mean"] for r in macro if r["split"] == primary_split}
    if ("v2_M1", "gain_zero_mse") in mv:
        findings.append(f"v2 M1의 gain MSE는 {mv['v2_M1', 'gain_mse']:.5f}, 항상 0-gain 기준은 {mv['v2_M1', 'gain_zero_mse']:.5f}입니다. 예측 이득−관측 이득의 macro 평균은 {100 * mv['v2_M1', 'gain_bias']:+.2f} pp입니다. 평균뿐 아니라 cell·horizon별 편차를 함께 확인해야 합니다.")
    findings.append("v1은 현재 posterior를 직접 평가할 수 있지만 native 미래 예측분포가 없어 미래 예측·이득 calibration 순위를 v2와 직접 매길 수 없습니다.")
    archived = [e.get("archive_reproduction", {}).get("status") for c in cells for s in c["splits"].values() for e in s["models"].values() if "archive_reproduction" in e]
    text = ["# PGM 자체 평가: v1 · v2", "", "새 PGM 학습이나 stopping threshold 선택 없이, 기존 frozen bank와 공개된 모델로 계산한 결과입니다.", "",
            f"대상: {len(cells)}개 공통 cell. 주 결과: {primary_split}. Prefix={protocol['prefixes']}, horizon={protocol['horizons']}, Monte Carlo={protocol['simulations']}회.",
            "문제마다 prefix를 동일하게 가중하고, 미래 평가는 bank 안에 들어오는 prefix/horizon 조합을 동일하게 가중합니다. Cell 간 평균은 문제 수와 무관한 macro 평균입니다.",
            "95% CI는 문제 단위 paired bootstrap입니다. Prefix·horizon을 별도 독립 표본으로 세지 않습니다. Macro 표에는 유의성 검정이나 CI를 부여하지 않습니다.", "",
            "## 평가 가능 범위", "",
            table(["모델", "현재 posterior", "미래 예측", "기대 이득 calibration"], [
                ["v1 mixture / raw", "가능", "N/A: 미래 관측 생성분포 없음", "N/A: posterior만으로 식별 불가"],
                ["v2 M1, θ 상수 CRP, M2 confidence", "가능", "답의 native 주변 예측분포 가능", "고정 다수결의 horizon별 이득 평가 가능"],
                ["v2 copy 확장", "옵션으로 가능", "현재 adapter에서는 N/A", "미래 similarity 분포 필요"],
                ["v2 flat-wrong", "정규화한 posterior의 진단만 가능", "N/A: 정규화된 관측 생성분포 아님", "N/A"]]), "",
            "## 주 비교", "", "아래 loss는 모두 낮을수록 좋습니다. Unique NLL은 bank 전체에서 정답 판정 후보가 최대 하나인 공통 부분집합만 사용합니다.",
            "Reference Brier는 두 모델에 동일한 다수결 답을 평가시킨 값입니다. Gain MSE−zero가 음수이면 추가 이득을 항상 0으로 예측하는 기준보다 좋습니다.", "",
            table(["모델", "Unique NLL", "Reference Brier", "OTHER Brier", "다음 답 NLL", "Gain MSE", "Gain MSE−zero"], macro_rows), "",
            "## 관측된 패턴", "", *[p for f in findings for p in (f, "")],
            "아래 posterior 그림은 데이터셋별 세 모델의 평균입니다.", "",
            "![Posterior quality](figures/posterior_quality.png)", "",
            "## 같은 답에 부여한 정답 확률: cell별 비교", "", "Δ는 v2 M1 − calibrated v1입니다. 음수는 v2의 Brier loss가 더 낮다는 뜻입니다.", "",
            table(["Cell", "v1 Brier", "v2 Brier", "Δ", "paired 95% CI"], comparison_rows), "",
            "## v2 미래 예측과 기대 이득", "", "예측 이득과 관측 이득은 정확도 단위입니다. 0.01은 1 pp입니다. 미래 답 선택도 동일한 다수결과 최초 관측 동률 규칙을 사용합니다.", "",
            table(["Cell", "모델", "다음 답 NLL", "예측 이득", "관측 이득", "Gain MSE−zero"], future_rows), "",
            "아래 calibration 그림은 데이터셋별 모델·prefix·horizon의 같은 예측 구간을 합칩니다. 점이 클수록 포함된 관측 상태가 많습니다. 이 상태들은 독립 표본이 아니며, CI는 별도로 문제 단위에서 계산합니다.", "",
            "![Gain calibration](figures/gain_calibration.png)", "",
            "## 라벨과 범위 점검", "", table(["Cell", "문제 수", "정답 후보 여러 개", "16개 안에 정답 없음", "빈 응답"], audit_rows), "",
            "## 해석의 한계", "",
            "1. 주 라벨은 xVerify가 추출된 답 후보에 부여한 정오입니다. 증명 전체의 타당성이나 인간 검수 정답을 보장하지 않습니다.",
            "2. Accepted-set NLL은 여러 정답 판정 후보의 확률을 합친 기술 통계입니다. 문자열 동치류가 확정되지 않아 전체 multiclass strictly proper score라고 해석하지 않습니다. Unique NLL/Brier는 모호한 문제를 공통으로 제외한 결과이며, 전체 문제의 성능을 대신하지 않습니다.",
            "3. v1은 기존 calibration split에서 학습한 temperature를 사용하고 raw 결과도 함께 제공합니다. v2는 공개 posterior 그대로입니다. 입력 feature·학습 target·학습 목적이 달라 순수한 그래프 구조의 인과 효과 비교가 아닙니다. Dev는 v1 early stopping에 사용되었으므로 held-out 주 평가로 해석하지 않습니다.",
            "4. v1의 미래 예측·이득을 0으로 간주하지 않습니다. N/A는 모델이 그 분포를 제공하지 않는다는 뜻입니다. 추가 readout을 학습하면 별도 모델로 구분해야 합니다.",
            "5. v2 미래 평가는 단일 정답·CRP 가정 아래 답 ID를 생성합니다. 빈 답을 생성하는 모형이 없어 INVALID 확률은 0입니다. 관측된 0확률 사건 수와 1e-12 clipping을 보고합니다. M2의 미래 confidence는 고정 다수결에 사용되지 않아 정확히 주변화할 수 있습니다. 반면 copy에는 미래 similarity 분포가 필요하므로 이를 생략한 근사를 native 예측이라고 부르지 않습니다.",
            "6. 기대 이득은 max/clipping 전의 horizon별 signed gain입니다. 기존 보고서의 posterior 동률 처리·max-horizon VOI나 실제 stopping policy 성능과 동일하지 않습니다. 토큰 비용이나 PGM 계산 시간도 이 확률 품질 평가에는 들어가지 않습니다.",
            "7. 미래 suffix는 문제마다 한 번 관측한 표본입니다. 개별 문제의 실제 기대 VOI를 안다고 해석하지 않습니다. Monte Carlo SE를 별도 저장하고 MSE의 MC 분산 보정값도 제공하지만, bootstrap CI는 response-bank 재생성·judge 오류·학습 seed의 불확실성을 포함하지 않습니다.",
            "8. 이 test bank는 기존 연구에서 이미 관찰한 데이터이므로 탐색적 비교입니다. 앞으로 PGM 구조를 선택한 뒤 주장할 최종 성능에는 새 holdout이 필요합니다.", "",
            "확률 loss나 이득 calibration이 좋아지는 것과 실제 stopping 효용이 좋아지는 것은 별도로 검증해야 합니다. 특히 0-gain 기준은 답이 거의 바뀌지 않는 bank에서 강한 기준이며, 이를 못 이겼다는 사실만으로 VOI 중단의 효용을 부정할 수는 없습니다.", "",
            "## 재현 및 산출물", "",
            f"기존 v1 fixed-prefix NLL 재현 검사: {archived.count('passed')}개가 2e-5 허용오차 내 일치. {archived.count('minor_nll_difference_same_accuracy')}개는 정확도가 모두 같지만 NLL에 미세한 차이가 있어 별도 표시합니다. 차이의 원인은 확정하지 않았고 cells JSON에 모든 차이를 기록합니다. 1e-3을 넘는 NLL 차이나 fixed-prefix 정확도 차이는 실행을 중단합니다. 원래 보고서를 덮어쓰지 않습니다.",
            "pgm diagnose prepare-v1 --workers 3", "pgm diagnose run --workers 2", "pgm diagnose report", "",
            "summary.csv: cell·split·모델별 평균과 CI. posterior_by_prefix.csv: 시점별 posterior 점수. future_by_horizon.csv: 미래 사건과 signed gain 점수.",
            "posterior_calibration.csv / gain_calibration.csv: reliability bin. paired_contrasts.csv: 문제별 paired 차이. capabilities.csv: 지원 범위. cells/*.json: 전체 점수·설정·자료 hash.",
            "artifacts/diagnostics/: 원래 모델을 다시 학습하지 않고 재사용할 posterior·예측 cache. v1 정책·체크포인트는 manifest SHA256 검증, 원격 bank·embedding 부분 읽기는 revision 고정과 compact bank의 ID·라벨 대조를 사용하며 원본 전체 파일 hash 검증은 하지 않습니다."]
    markdown = "\n".join(text) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(markdown)
    figures(out, cells)
    body = markdown_html(markdown)
    (out / "report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                                    "<title>PGM 자체 평가</title><style>body{font-family:system-ui,sans-serif;max-width:1280px;margin:40px auto;padding:0 24px;line-height:1.7;color:#17212b;background:#fafbfc}h1,h2{line-height:1.25;margin-top:2em}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px;background:white}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce2e8}th{background:#eaf0f7}img{width:100%;margin:20px 0}.scroll{overflow:auto}p{max-width:1100px}</style><body>" + body + "</body></html>")
    return out / "report.html"
