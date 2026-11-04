"""results/report.html: a self-contained page with the PGM structure, setup and results."""
from __future__ import annotations

import json
from pathlib import Path

from pgm_asc.config import RESULTS, cells
from pgm_asc.report.figures import load_evaluations

TEMPLATE = Path(__file__).with_name("report_template.html")


def page_data(evaluations):
    out = []
    for cell in cells():
        e = evaluations.get(cell.name)
        if e is None:
            continue
        methods = {k.replace("_s42", ""): [round(v["correct"] * 100, 2), round(v["mean_output_tokens"]), round(v["samples"], 2)]
                   for k, v in e["summary"].items() if "_s43" not in k and "_s44" not in k}
        contrast = lambda c, p: [round(c["accuracy_difference"] * 100, 2), [round(x * 100, 2) for x in c["accuracy_difference_ci95"]],
                                 round(c["output_token_ratio"], 3), round(p, 4)]
        stability = e["seed_stability"]["mixture__selected"]
        out.append({"key": cell.name, "section": "tr" if cell.transfer else "in", "model": cell.model, "dataset": cell.task,
                    "n": e["questions"], "m": methods,
                    "pc": {k: contrast(c, c["holm_p_four_primary_structure_contrasts"]) for k, c in e["primary_structure_contrasts"].items()},
                    "sc": {k: contrast(c, c["mcnemar_exact_p"]) for k, c in e["baseline_contrasts"].items()},
                    "seed": [round(stability["correct"]["mean"] * 100, 2), round(stability["correct"]["std_across_3_seeds"] * 100, 2),
                             round(stability["mean_output_tokens"]["mean"])]})
    return out


def make_report(path=RESULTS / "report.html"):
    data = json.dumps(page_data(load_evaluations()), separators=(",", ":"))
    path.write_text(TEMPLATE.read_text().replace("__DATA__", data))
