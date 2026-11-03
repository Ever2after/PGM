import pytest
import numpy as np
import torch
from pgm_asc.core.answers import answer_groups
from pgm_asc.core.answers import boxed_span
from pgm_asc.core.answers import equivalent
from pgm_asc.core.io import sample_seed
from pgm_asc.data.features import prefix_features
from pgm_asc.collection.token_stats import raw_statistics

def test_nested_last_complete_box_and_unclosed_tail():
    s = r"Earlier \boxed{1}. Final \boxed{\frac{1}{\sqrt{2}}}. Unfinished \boxed{"
    box = boxed_span(s)
    assert box["answer"] == r"\frac{1}{\sqrt{2}}"
    assert s[box["body_start"]:box["body_end"]] == box["answer"]
    assert boxed_span("no answer") is None
    assert boxed_span(r"The answer is $\boxed 2$.")["answer"] == "2"

def test_math_equivalence_and_online_grouping_do_not_need_gold():
    assert equivalent(r"\frac{1}{2}", "0.5")
    assert not equivalent("1", "2")
    assert not equivalent(None, None)
    reps, groups = answer_groups(["0.5", r"\frac{1}{2}", "2", None])
    assert groups == [0, 0, 1, -1]
    assert reps == ["0.5", "2"]

def record(answer):
    return {"answer": answer, "reasoning_mean_logprob": -1.0, "answer_mean_logprob": -0.5, "reasoning_mean_entropy": 2.0, "generated_token_count": 10, "truncated": False}

def test_prefix_invariance_ties_and_missing_answer():
    records = [record("1"), record("2"), record("2")]
    vectors = np.array([[1., 0.], [0., 1.], [0., 1.]])
    a = prefix_features(records[:2], vectors[:2])
    assert a["leader_answer"] == "1"  # first-observed tie, independent of gold
    assert prefix_features(records, vectors)["leader_answer"] == "2"
    again = prefix_features(records[:2], vectors[:2])
    np.testing.assert_array_equal(a["relation"], again["relation"])
    missing = prefix_features([record(None), record(None)], vectors[:2])
    assert missing["leader_answer"] is None
    assert np.isfinite(missing["confidence"]).all()

def test_raw_confidence_is_original_distribution():
    logits = torch.tensor([[0., 1., 2.]])
    tokens = torch.tensor([2])
    logp, entropy = raw_statistics(logits, tokens)
    p = torch.softmax(logits, dim=-1)
    assert logp.item() == pytest.approx(torch.log(p[0, 2]).item())
    assert entropy.item() == pytest.approx(-(p * torch.log(p)).sum().item())

def test_response_seed_reproducible_and_distinct():
    assert sample_seed(42, "q1", 0) == sample_seed(42, "q1", 0)
    assert len({sample_seed(42, "q1", i) for i in range(16)}) == 16
