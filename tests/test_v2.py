"""v2 model on data sampled from its own generative story (no artifacts needed)."""
import numpy as np
import pytest

from pgm_asc.baselines import decide
from pgm_asc.v2 import OTHER
from pgm_asc.v2 import baselines as B
from pgm_asc.v2.model import Spec, fit, posterior, readable, transforms
from pgm_asc.v2.run import arm_family, curve
from pgm_asc.v2.voi import voi


def sample_split(rng, q, theta=1.5, a=.6, b=.4, n=16):
    ids = np.zeros((q, n), np.int8)
    correct = np.zeros((q, 16), bool)
    x = np.zeros((q, n, 3), np.float32)
    for i in range(q):
        acc = rng.beta(a, b)
        labels, wrong = [], {}
        for j in range(n):
            z = rng.random() < acc
            if z:
                lab = "c"
            else:
                m = sum(wrong.values())
                if rng.random() < theta / (m + theta):
                    lab = f"w{len(wrong)}"
                    wrong[lab] = 0
                else:
                    keys = list(wrong)
                    lab = keys[rng.choice(len(keys), p=np.array([wrong[k] for k in keys]) / m)]
                wrong[lab] += 1
            labels.append(lab)
            x[i, j, 0] = rng.normal(.8 * z + .6 * (acc - .5), 1)
        reps = list(dict.fromkeys(labels))
        ids[i] = [reps.index(v) for v in labels]
        correct[i, :len(reps)] = [r == "c" for r in reps]
    x[..., 1] = x[..., 0] + rng.normal(0, .1, x[..., 0].shape)
    x[..., 2] = 5.
    sim = rng.random((q, n, n)).astype(np.float32)
    sim = (sim + sim.transpose(0, 2, 1)) / 2
    return {"ids": ids, "valid": np.ones((q, n), bool), "correct": correct, "raw_correct": correct[np.arange(q)[:, None], ids],
            "x": x, "tokens": rng.integers(100, 400, (q, n)).astype(np.int32), "sim_text": sim, "sim_hidden": sim,
            "question_id": np.array([f"q{i}" for i in range(q)])}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return {s: sample_split(rng, n) for s, n in [("calibration", 40), ("train", 400), ("dev", 30), ("test", 30)]}


def test_fit_recovers_generative_parameters(data):
    spec = Spec(theta="const", wrong="crp", x="both", copy="none", other=True, stop="confidence")
    tf = transforms(data["train"], spec, [0])
    params, summary = fit(data["train"], spec, tf)
    assert summary["nll_per_question"] < summary["nll_per_question_start"]
    v = readable(params)
    assert abs(v["theta(q=0.5)"] - 1.5) < .6
    assert abs(v["a"] - .6) < .25 and abs(v["b"] - .4) < .2
    assert float(params["b2"][0]) > .3          # standardized evidence shift (true .8 before scaling)


def test_posterior_is_normalized_and_calibrated(data):
    spec = Spec(theta="q", wrong="crp", x="both", copy="text", other=True, stop="voi", decision="map")
    tf = transforms(data["train"], spec, [0])
    params, _ = fit(data["train"], spec, tf)
    p = posterior(params, spec, data["dev"], tf)
    assert np.allclose(p.post.sum(2), 1)
    assert ((p.post > 0) <= p.allowed).all()
    # unseen slots are never chosen, and the chosen answer at t=16 is mostly the correct one
    rows = np.arange(len(p.chosen))
    assert (p.chosen[:, 0] == 0).all()
    hit = data["dev"]["correct"][rows, p.chosen[:, 15]]
    assert hit.mean() >= .6
    v = voi(params, spec, data["dev"], p, key="test")
    assert v.shape == (len(p.chosen), 15) and (v >= -.02).all()
    family = arm_family({"chosen": p.chosen, "confidence": p.confidence, "other": p.post[:, :, OTHER], "voi": v}, spec)
    pts = curve(data["dev"], family)
    # a larger lambda never samples more
    samples = [pt["mean_samples"] for pt in pts]
    assert all(a <= b + 1e-9 for a, b in zip(samples, samples[1:]))


def test_baselines_match_v1(data):
    fams = B.families(data, "dev")
    ids = data["dev"]["ids"].astype(int)
    scores = data["dev"]["x"][..., 0].astype(float)
    for i in range(len(ids)):
        for method, key, fam in [("sc16", 16, "sc"), ("asc", .95, "asc"), ("esc", 4, "esc")]:
            stop, chosen, _ = decide(list(ids[i]), list(scores[i]), method)
            assert fams[fam][key][0][i] == stop and fams[fam][key][1][i] == chosen


def test_pipeline_on_synthetic_bank(study):
    """prepare -> run on the tiny synthetic bank of conftest; nothing is written inside the repository."""
    import importlib
    import pgm_asc.v2 as v2
    importlib.reload(v2)
    for name in ("pgm_asc.v2.data", "pgm_asc.v2.model", "pgm_asc.v2.voi", "pgm_asc.v2.baselines", "pgm_asc.v2.run"):
        importlib.reload(importlib.import_module(name))
    from pgm_asc.baselines import sc_accuracy
    from pgm_asc.config import ROOT
    from pgm_asc.data.bank import load_bank
    from pgm_asc.utils import read_json
    from pgm_asc.v2.data import load, prepare
    from pgm_asc.v2.run import run
    prepare("qwen3b", "math")
    data = load("qwen3b", "math")
    assert data["test"]["ids"].shape == (8, 16) and data["dev"]["sim_hidden"].shape == (6, 16, 16)
    run(["qwen3b"], ["math"], workers=2)
    path = v2.result_path("qwen3b", "math")
    assert ROOT not in path.parents
    result = read_json(path)
    sc16 = result["sc16"]["test"]["accuracy"]
    assert abs(sc16 - sc_accuracy(load_bank("qwen3b", "math", "test", embeddings=False))) < 1e-12
    assert set(result["selected"]["accuracy"]) >= set(v2.CONFIG["arms"]) | {"sc", "asc", "esc", "reasc_offline", "reasc_online"}
    from pgm_asc.config import RESULTS
    from pgm_asc.v2.report import report
    importlib.reload(importlib.import_module("pgm_asc.v2.report"))
    from pgm_asc.v2.report import report
    report()
    assert (RESULTS / "v2" / "README.md").exists() and (RESULTS / "v2" / "figures" / "ladder.png").exists()


def test_new_method_plugin_and_csv(study):
    """A user method (plugin and CSV) is scored by the same protocol and merged into the report."""
    import csv
    import importlib
    from pgm_asc.config import ROOT, RESULTS
    from pgm_asc.utils import read_json
    importlib.reload(importlib.import_module("pgm_asc.v2.methods"))
    importlib.reload(importlib.import_module("pgm_asc.v2.report"))
    from pgm_asc.v2 import baselines as B
    from pgm_asc.v2.data import load
    from pgm_asc.v2.methods import REGISTRY, csv_method, evaluate, load_plugin
    from pgm_asc.v2.report import plot, report
    names = load_plugin(str(ROOT / "examples" / "plurality_margin.py"))
    assert names == ["plurality_margin"]
    paths = evaluate(REGISTRY["plurality_margin"], ["qwen3b"], ["math"])
    assert len(paths) == 1 and ROOT not in paths[0].parents
    r = read_json(paths[0])
    assert set(r["curves"]["test"]) == {"sc", "M1", "plurality_margin"} and "plurality_margin" in r["selected"]["default"]
    # CSV path: SC@k written as decisions must reproduce the SC@k curve exactly
    data = load("qwen3b", "math")
    path = RESULTS / "sc_decisions.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "split", "question_id", "threshold", "samples", "response_index"])
        for split in ("dev", "test"):
            fam = B.sc_family(data[split])
            for k, (stop, chosen) in fam.items():
                ids = data[split]["ids"]
                for i, qid in enumerate(data[split]["question_id"]):
                    j = int(np.flatnonzero(ids[i] == chosen[i])[0])
                    w.writerow(["qwen3b", "math", split, qid, k, int(stop[i]) + 1, j])
    evaluate(csv_method(str(path), "sc_csv", "SC (CSV)"), ["qwen3b"], ["math"])
    rc = read_json(RESULTS / "v2" / "methods" / "sc_csv" / "qwen3b_math.json")
    strip = lambda pts: sorted((p["threshold"], p["accuracy"], p["mean_tokens"]) for p in pts)
    assert strip(rc["curves"]["test"]["sc_csv"]) == strip(rc["curves"]["test"]["sc"])
    report()
    html = (RESULTS / "v2" / "report.html").read_text(encoding="utf-8")
    assert '"extra":["plurality_margin","sc_csv"]' in html
    plot(["M1", "plurality_margin"], RESULTS / "v2" / "figures" / "custom")
    assert (RESULTS / "v2" / "figures" / "custom.png").exists()

