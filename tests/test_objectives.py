"""The objective ablation must change losses, not the M1 probability model."""
import numpy as np
import pytest
import torch

from pgm_asc.objectives.train import Counts, Objective, SPEC, common_training_subset, fit_one, prefix_weights
from pgm_asc.v2.model import Grid, factors, init_params, log_prior, make_batch, neg_log_likelihood, target_hypothesis


def small_bank():
    ids = np.array([[0, 1, 0, 2] * 4, [0, 1] * 8, [0] * 16, list(range(16))])
    valid = np.ones_like(ids, bool)
    valid[0, [0, 3, 5]] = False
    correct = np.zeros((4, 16), bool)
    correct[0, 1] = True
    correct[1, 0] = True
    return {"ids": ids, "valid": valid, "correct": correct,
            "question_id": np.array(["a", "b", "c", "d"])}


def test_compressed_likelihood_and_gradients_match_sequential_m1_with_invalid_answers():
    arrays, grid = small_bank(), Grid()
    counts = Counts(arrays)
    params = init_params(SPEC, 0)
    with torch.no_grad():
        for key, value in zip(params, [.2, -.4, .7, -.3]):
            params[key].fill_(value)
    batch = make_batch(arrays, SPEC, {})
    joint, prefix = counts.losses(params, grid)
    native = neg_log_likelihood(params, SPEC, batch, torch.tensor(target_hypothesis(arrays)), grid) / 16
    torch.testing.assert_close(joint, native, rtol=1e-10, atol=1e-10)
    j2, _ = counts.losses(params, grid, need_prefix=False)
    torch.testing.assert_close(joint, j2)
    # Native conditional likelihood, retaining the original sequential factors.
    hyp = torch.arange(17).expand(len(arrays["ids"]), -1)
    lj = factors(params, SPEC, batch, hyp, grid).cumsum(1) + log_prior(params, grid)
    allowed = torch.cat([batch.seen, torch.ones((4, 16, 1), dtype=torch.bool)], 2)
    lp = torch.logsumexp(lj, -1).masked_fill(~allowed, -torch.inf).log_softmax(-1)
    target = np.where(np.take_along_axis(arrays["correct"], arrays["ids"], 1) & arrays["valid"], arrays["ids"], 16)
    seen_true = np.cumsum(np.eye(16)[arrays["ids"]] * arrays["valid"][..., None], 1) * arrays["correct"][:, None]
    target = np.where(seen_true.any(2), seen_true.argmax(2), 16)
    native_prefix = -lp.gather(2, torch.tensor(target)[..., None]).mean()
    torch.testing.assert_close(prefix, native_prefix, rtol=1e-10, atol=1e-10)
    for loss_a, loss_b in [(joint, native), (prefix, native_prefix)]:
        ga = torch.autograd.grad(loss_a, list(params.values()), retain_graph=True)
        gb = torch.autograd.grad(loss_b, list(params.values()), retain_graph=True)
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-9)


def test_training_mask_and_weightings_are_fixed_across_objectives():
    arrays = small_bank()
    arrays["correct"][0, 0] = True
    sub, keep = common_training_subset(arrays)
    assert keep.tolist() == [False, True, True, True]
    assert len(sub["ids"]) == 3
    for name in ("uniform", "v1"):
        assert prefix_weights(name).sum() == pytest.approx(1.)
    np.testing.assert_array_equal(prefix_weights("v1"), [.2, .2, .1, .1] + [.05] * 4 + [.025] * 8)


@pytest.mark.parametrize("kind,mixing", [("joint", None), ("prefix", None), ("mixed", 1.)])
def test_each_training_mode_preserves_four_parameters_and_improves_its_loss(kind, mixing):
    from pgm_asc.objectives import CONFIG
    from pgm_asc.objectives.train import data_loss
    settings = {**CONFIG["training"], "max_iter": 30}
    counts, grid = Counts(small_bank()), Grid()
    objective = Objective(kind, "uniform", mixing)
    p = init_params(SPEC, 0)
    j, c = counts.losses(p, grid, need_prefix=kind != "joint")
    before = float(data_loss(j, c, objective).detach())
    result = fit_one(counts, objective, settings, 0)
    assert result["parameters"] == 4 and len(result["params"]) == 4
    assert result["data_loss"] < before


def test_paired_forecast_preserves_future_probabilities_and_scores_repair_damage():
    from pgm_asc.diagnostics.predictive import forecast
    from pgm_asc.diagnostics.metrics import future_scores
    from pgm_asc.v2.model import posterior, log_theta
    arrays, grid = small_bank(), Grid()
    params = init_params(SPEC, 0)
    obs = {k: v for k, v in arrays.items() if k != "correct"}
    p = posterior(params, SPEC, obs, {})
    theta = log_theta(params, SPEC, grid).detach().exp().numpy()
    old = forecast(obs, p, grid.q.numpy(), theta, [1, 2], [1, 2], 32, 7)
    new = forecast(obs, p, grid.q.numpy(), theta, [1, 2], [1, 2], 32, 7,
                   paired_gain=True, transition_details=True)
    np.testing.assert_array_equal(new["future_reference_correct"], old["future_reference_correct"])
    np.testing.assert_array_equal(new["next_answer"], old["next_answer"])
    np.testing.assert_allclose(new["gain"], new["repair"]-new["damage"])
    assert np.all(new["gain"][arrays["valid"][:, 0], 0, 0] == 0)
    rows, _, loss = future_scores(new, arrays, [1, 2], [1, 2])
    assert "repair_brier" in loss and "damage_brier" in loss
    assert len([r for r in rows if r["event"] == "repair"]) == 4


def test_multiple_mc_seeds_average_means_and_propagate_standard_errors():
    from pgm_asc.objectives.run import average_forecasts
    draws = [{"repair": np.array([x]), "repair_mc_se": np.array([s])} for x, s in [(0., .3), (1., .4)]]
    averaged = average_forecasts(draws)
    assert averaged["repair"][0] == .5
    assert averaged["repair_mc_se"][0] == pytest.approx(.25)


def test_complete_objective_pipeline_locks_dev_choices_and_reuses_fits(tmp_path, monkeypatch):
    import copy
    import json
    from types import SimpleNamespace
    from pgm_asc import config
    from pgm_asc.objectives import CONFIG
    from pgm_asc.objectives import run, report
    from pgm_asc.v2.model import to_json
    artifacts, results = tmp_path / "artifacts", tmp_path / "results"
    (artifacts / "v2").mkdir(parents=True)
    (results / "v2" / "cells").mkdir(parents=True)
    study = copy.deepcopy(CONFIG)
    study["training"].update(max_iter=6, restart_seeds=[0], lambda_grid=[1.], weightings=["uniform"])
    study["evaluation"].update(splits=["test"], prefixes=[1, 2], horizons=[1, 2],
                               simulations_per_seed=8, mc_seeds=[5], bootstrap_replicates=20)
    arrays = small_bank()
    packed = {}
    for split in ("train", "dev", "test"):
        for key, value in arrays.items():
            packed[f"{split}.{key}"] = np.array([f"{split}-{q}" for q in value]) if key == "question_id" else value
    np.savez(artifacts / "v2" / "tiny_math.npz", **packed)
    (results / "v2" / "cells" / "tiny_math.json").write_text(json.dumps({"fitted": {"M0": {"params": to_json(init_params(SPEC, 0))}}}))
    fake = SimpleNamespace(ARTIFACTS=artifacts, RESULTS=results, ROOT=config.ROOT)
    monkeypatch.setattr(run, "config", fake)
    monkeypatch.setattr(report, "config", fake)
    out = results / "objective_ablation"
    run.run(["tiny"], ["math"], workers=1, out=out, study=study)
    lock = json.loads((out / "selection" / "tiny_math.json").read_text())
    r = json.loads((out / "cells" / "tiny_math.json").read_text())
    assert lock["selection_locked_at"] <= r["evaluation_started_at"]
    assert all(q.startswith("train-") for q in lock["train_question_ids"])
    assert set(r["splits"]["test"]["models"]) == {"joint", "prefix_uniform", "mixed_uniform", "released_M1"}
    assert all(v["count"] == 4 for v in r["parameters"].values())
    assert (out / "report.html").is_file() and (out / "paired_contrasts.csv").is_file()
    with np.load(r["prediction_cache"]["path"], allow_pickle=False) as z:
        for name in r["parameters"]:
            np.testing.assert_allclose(z[f"test.{name}.future.gain"],
                                       z[f"test.{name}.future.repair"]-z[f"test.{name}.future.damage"])
    # A resume must not re-fit or select hyperparameters from test results.
    monkeypatch.setattr(run, "fit_one", lambda *a, **k: pytest.fail("cached run must not refit"))
    run.run(["tiny"], ["math"], workers=1, out=out, study=study)
    assert json.loads((out / "selection" / "tiny_math.json").read_text()) == lock
