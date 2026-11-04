import numpy as np
import pytest
import torch

from pgm_asc.baselines import decide, offline_calibrate, trace
from pgm_asc.collect.extraction import boxed_span, confidence_groups, prediction
from pgm_asc.data.bank import exact_groups
from pgm_asc.data.partitions import OnlinePartitions
from pgm_asc.model import ARMS, PGM, make
from pgm_asc.policies import calibrate, select_candidate, stopping_index
from pgm_asc.stats import holm


def test_exact_groups_first_observed_order():
    reps, ids = exact_groups(["3", "4", "3", "", "4"])
    assert reps == ["3", "4", ""]
    assert ids.tolist() == [0, 1, 0, 2, 1]


def test_answer_extraction():
    assert prediction(r"so \boxed{\frac{1}{2}}", "") == r"\frac{1}{2}"
    assert prediction("no box", r"\boxed{ 7 }") == "7"
    assert prediction("x", "The answer is 1,234", "gsm8k") == "1234"
    assert boxed_span(r"a \boxed{3} then \boxed{4}")["answer"] == "4"


def test_confidence_groups_windows():
    scores = np.r_[np.ones(200), np.zeros(56)]
    groups = confidence_groups(scores, window=128)
    assert groups["lowest"] == 0.0 and groups["bottom10"] < groups["lowest"] + 1e-9 + 0.5


def test_baselines():
    preds = ["1", "1", "1", "1", "2", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]
    assert decide(preds, [0.] * 16, "sc16")[:2] == (15, 0)
    assert decide(preds, [0.] * 16, "esc")[0] == 3
    asc = trace(preds, [0.] * 16, "asc")
    assert asc["confidence"][-1] > 0.95
    cal = offline_calibrate([1., 2., 3., 4.], [False, False, True, True])
    assert cal["gate"] >= 3.0


def test_partitions_sizes_and_random_control():
    rng = np.random.default_rng(0)
    parts = OnlinePartitions([0.1, 0.4, 0.7, 0.9], seed=1)
    for _ in range(6):
        snap = parts.update(rng.normal(size=8))
    random = parts.snapshot(random=True)
    assert snap["sizes"].shape == (6, 6)
    assert (snap["sizes"][0] == 6).all() and (snap["sizes"][5] == 1).all()
    for m in range(6):  # matched cluster-size multisets
        assert sorted(snap["sizes"][m]) == sorted(random["sizes"][m])


def test_heads_are_distributions():
    b, n = 3, 32
    batch = {"x": torch.randn(b, n, 9), "ids": torch.full((b, n), -1), "response_mask": torch.zeros(b, n, dtype=torch.bool),
             "candidate_mask": torch.zeros(b, n + 1, dtype=torch.bool), "candidate": torch.randn(b, n + 1, 6), "w": torch.randn(b, 25),
             "sizes": torch.ones(b, 6, n), "gate": torch.randn(b, 6, 12), "partition_mask": torch.ones(b, 6, dtype=torch.bool)}
    batch["ids"][:, :3] = torch.tensor([0, 1, 0])
    batch["response_mask"][:, :3] = True
    batch["candidate_mask"][:, [0, 1, n]] = True
    batch.update(random_sizes=batch["sizes"], random_gate=batch["gate"], random_partition_mask=batch["partition_mask"])
    for name, spec in ARMS.items():
        if spec.get("use_embeddings"):
            continue
        lp = make(spec, 8)(batch)
        assert torch.allclose(lp.exp().sum(-1), torch.ones(b), atol=1e-5), name
    assert isinstance(make(ARMS["mixture"], 8), PGM)


def test_stopping_rule():
    history = {"chosen": np.zeros(16, int), "confidence": np.r_[0.99, 0.5, 0.97, np.full(13, 0.99)], "can_stop": np.ones(16, bool)}
    policy = {"minimum": 2, "patience": 1, "cap": 16}
    assert stopping_index(history, policy | {"gamma": 0.95}) == (2, "confidence")
    assert stopping_index(history, policy | {"gamma": None}) == (15, "cap")


def test_threshold_selection_and_calibration():
    rows = [{"accuracy": 0.80, "mean_tokens": 100, "mean_samples": 3}, {"accuracy": 0.85, "mean_tokens": 300, "mean_samples": 8},
            {"accuracy": 0.70, "mean_tokens": 50, "mean_samples": 2}]
    assert select_candidate(rows, sc=0.81)["mean_tokens"] == 100
    lp = calibrate(np.log([[0.7, 0.2, 0.1]]), {"kind": "global", "T": 2.0})
    assert np.isclose(np.exp(lp).sum(), 1.0)


def test_holm_monotone():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03) and adjusted["b"] == pytest.approx(0.06)
