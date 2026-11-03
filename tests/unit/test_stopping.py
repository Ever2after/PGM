import copy

import numpy as np
import torch

from pgm_asc.core.integrity import problem_key
from pgm_asc.data.prefixes import bucket_ids
from pgm_asc.data.prefixes import prefix_weights
from pgm_asc.policies.calibration import calibrate
from pgm_asc.policies.stopping import stopping_index
from pgm_asc.evaluation.replay import replay_policy
from pgm_asc.policies.stopping import select_candidate
from pgm_asc.policies.continuation import continuation_features
from pgm_asc.training.continuation import continuation_targets
from pgm_asc.evaluation.replay import sc_accuracy


def test_prefix_buckets_have_equal_total_objective_weight():
    times = np.tile(np.arange(1, 33), 3)
    weights = prefix_weights(times)
    ids = bucket_ids(times)
    assert np.allclose([weights[ids == k].sum() for k in range(6)], 1 / 6)
    assert bucket_ids([1, 2, 3, 4, 5, 8, 9, 16, 17, 32]).tolist() == [0, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_bucket_identity_and_other_bias_preserve_observed_answer_order():
    lp = np.log([[.6, .3, .1], [.2, .7, .1]])
    identity = {"kind": "bucket", "log_T": [0.] * 6, "bias": [0.] * 6}
    assert np.allclose(calibrate(lp, [1, 2], identity), lp)
    parameters = {**identity, "bias": [1.] * 6}
    new = calibrate(lp, [1, 2], parameters)
    assert np.allclose(np.exp(new).sum(1), 1.)
    assert np.all(np.exp(new[:, -1]) > np.exp(lp[:, -1]))
    assert np.array_equal(new[:, :-1].argmax(1), lp[:, :-1].argmax(1))


def toy():
    history = {"chosen": np.r_[0, np.ones(31, dtype=int)],
               "confidence": np.full(32, .9), "can_stop": np.ones(32, bool)}
    data = {"gold_groups": torch.zeros(32, 33, dtype=torch.bool),
            "w": torch.zeros(32, 25), "candidate_mask": torch.ones(32, 33, dtype=torch.bool)}
    data["gold_groups"][:, 1] = True
    metadata = [{"question_id": "q", "token_cost": (i + 1) * 10, "leader": 1} for i in range(32)]
    return history, data, metadata


def test_stopping_is_causal_and_requires_answer_persistence():
    history, _, _ = toy()
    policy = {"minimum": 2, "patience": 2, "cap": 32, "gamma": .85}
    assert stopping_index(history, policy) == (2, "confidence")
    changed = copy.deepcopy(history)
    changed["chosen"][3:] = 8
    changed["confidence"][3:] = 0
    assert stopping_index(changed, policy) == stopping_index(history, policy)


def test_cap_uses_its_current_answer_even_with_no_early_stop():
    history, data, metadata = toy()
    history["chosen"][16:] = 0
    policy = {"minimum": 2, "patience": 1, "cap": 16, "gamma": None}
    result = replay_policy([history], data, metadata, policy)
    assert result["accuracy"] == 1.
    assert result["rows"][0]["chosen"] == 1
    assert result["mean_tokens"] == 160


def test_continuation_targets_include_both_corrections_and_regressions():
    history, data, metadata = toy()
    history["chosen"][8:] = 0
    targets = continuation_targets([history], data, metadata)
    assert targets[0, 0] == 1
    assert targets[4, 0] == -1
    assert targets[31, 1] == 0
    before = continuation_features([history], data)
    changed = copy.deepcopy(history)
    changed["chosen"][12:] = 9
    after = continuation_features([changed], data)
    assert np.array_equal(before[:12], after[:12])


def test_continuation_policy_can_stop_low_value_but_respects_minimum():
    history, _, _ = toy()
    history["can_stop"][:] = False
    policy = {"minimum": 4, "patience": 1, "cap": 16, "gamma": .99, "continuation_lambda": .00001}
    values = np.tile([-1., 10.], (32, 1))
    assert stopping_index(history, policy, values) == (3, "continuation_value")


def test_continuation_targets_never_use_responses_beyond_the_accepted_cap():
    history, data, metadata = toy()
    history['chosen'][8:] = 0
    target = continuation_targets([history], data, metadata, cap=8)
    assert target[6].tolist() == [0., 10.]
    assert np.all(target[7:] == 0.)
    changed = copy.deepcopy(history)
    changed['chosen'][8:] = 1
    other = continuation_targets([changed], data, metadata, cap=8)
    assert np.array_equal(target[:8], other[:8])


def test_stage_gate_rejects_cost_regression_and_repairs_infeasibility():
    previous = {"accuracy": .82, "mean_tokens": 100., "mean_samples": 4.}
    worse = {"accuracy": .83, "mean_tokens": 101., "mean_samples": 4.}
    selected, adopted = select_candidate([worse], .83, previous)
    assert not adopted and selected["mean_tokens"] == 100.
    fixed = {"accuracy": .83, "mean_tokens": 120., "mean_samples": 5.}
    selected, adopted = select_candidate([fixed], .84, previous)
    assert adopted and selected["accuracy"] == .83


def test_no_answer_does_not_count_other_as_a_correct_sc_prediction():
    _, data, metadata = toy()
    data["candidate_mask"][:, :32] = False
    data["gold_groups"][:, 32] = True
    for row in metadata:
        row["leader"] = 32
    assert sc_accuracy(data, metadata) == 0.


def test_problem_exclusion_normalizes_unicode_and_whitespace():
    assert problem_key("Ａ + B\n C") == problem_key("A + B C")


def test_dynamic_cges_support_maps_other_to_the_fixed_last_column():
    from pgm_asc.policies.baselines import cges_logp
    histories = [{"distributions": [np.array([1.])] + [np.array([.6, .3, .1])] * 31}]
    lp = cges_logp(histories)
    assert lp.shape == (32, 33)
    assert np.exp(lp[0, -1]) == 1.
    assert np.all(lp[0, :-1] < -1e8)
    assert np.allclose(np.exp(lp[1, [0, 1, 32]]), [.6, .3, .1])


def test_bucket_fit_float32_model_outputs_use_stable_float64_optimization():
    from pgm_asc.policies.calibration import _fit_bucket
    from scipy.special import log_softmax
    rng = np.random.default_rng(42)
    lp = log_softmax(rng.normal(size=(128, 4)) * 3., axis=1).astype(np.float32)
    target = rng.integers(0, 4, size=128)
    times = np.tile(np.arange(1, 33), 4)
    left = _fit_bucket(lp, target, times, .8, .1)
    right = _fit_bucket(lp.astype(np.float64), target, times, .8, .1)
    assert np.allclose(left['log_T'], right['log_T'], atol=1e-12)
    assert np.allclose(left['bias'], right['bias'], atol=1e-12)
