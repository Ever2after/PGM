import numpy as np
import torch
import torch.nn.functional as F

from pgm_asc.models.partitions import OnlinePartitions
from pgm_asc.models.selector import PGM


def batch():
    torch.manual_seed(7)
    n, b = 4, 2
    return {"x": torch.randn(b, n, 9), "w": torch.randn(b, 25),
            "ids": torch.tensor([[0, 0, 1, -1], [0, -1, -1, -1]]),
            "response_mask": torch.tensor([[True] * 4, [True, False, False, False]]),
            "candidate_mask": torch.tensor([[True, True, False, False, True], [True, False, False, False, True]]),
            "candidate": torch.randn(b, n + 1, 6),
            "sizes": torch.ones(b, 6, n), "random_sizes": torch.ones(b, 6, n),
            "gate": torch.randn(b, 6, 12), "random_gate": torch.randn(b, 6, 12),
            "partition_mask": torch.ones(b, 6, dtype=torch.bool),
            "random_partition_mask": torch.ones(b, 6, dtype=torch.bool)}


def test_alpha_one_partition_invariance_and_finite_gradient():
    data = batch()
    model = PGM(4, alpha=1.)
    a = model(data)
    data["sizes"] = torch.randint(1, 5, (2, 6, 4)).float()
    data["gate"] *= 100
    b = model(data)
    assert torch.allclose(a, b, atol=1e-6)
    F.nll_loss(b, torch.tensor([1, 4])).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_singletons_ignore_alpha_and_normalized_other_only():
    data = batch()
    model = PGM(4, alpha=0.)
    a = model(data)
    model.fixed_alpha = 1.
    assert torch.allclose(a, model(data), atol=1e-6)
    data["candidate_mask"][:, :4] = False
    data["ids"][:] = -1
    p = model(data).exp()
    assert torch.allclose(p.sum(1), torch.ones(2))
    assert torch.allclose(p[:, 4], torch.ones(2))


def test_causal_partitions_random_size_matching_and_dedup():
    vectors = np.array([[1., 0.], [1., 0.01], [0., 1.], [-1., 0.], [0., 0.]])
    state = OnlinePartitions([.1, .5, .9, .99], seed=8)
    prefix = None
    for i, v in enumerate(vectors):
        snapshot = state.update(v)
        for a, r in zip(state.assignments, state.random_assignments):
            assert sorted(np.bincount(a)) == sorted(np.bincount(r))
        if i == 2:
            prefix = [a.copy() for a in state.assignments]
    assert all(a[:3] == p for a, p in zip(state.assignments, prefix))
    assert snapshot["partition_mask"].sum() <= 6
    fresh = OnlinePartitions([.1, .5, .9, .99], seed=8)
    for v in vectors[:3]:
        fresh.update(v)
    assert fresh.assignments == prefix


def test_mixture_probability_matches_explicit_sum():
    data = batch()
    data["sizes"][:, 0] = 4
    model = PGM(4)
    mixture = model(data).exp()
    weights = model(data, diagnostics=True)[1]["weights"]
    pieces = []
    model.mode = "single"
    for k in range(6):
        model.partition = k
        pieces.append(model(data).exp())
    expected = (torch.stack(pieces, 1) * weights[:, :, None]).sum(1)
    assert torch.allclose(mixture, expected, atol=1e-6)


def test_synthetic_overfit_and_other_learning():
    torch.set_num_threads(2)
    data = batch()
    data["x"][0] += 3
    data["w"][1] -= 3
    model = PGM(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    target = torch.tensor([0, 4])
    before = float(F.nll_loss(model(data), target).detach())
    for _ in range(100):
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)
        loss.backward()
        optimizer.step()
    after = float(F.nll_loss(model(data), target).detach())
    assert after < before / 5
    assert torch.equal(model(data).argmax(1), target)


def test_constant_feature_scaling_does_not_explode_unseen_failures():
    from pgm_asc.data.banks import fit_scalers
    from pgm_asc.data.banks import scale
    data = batch()
    data["x"][:, :, 6:] = 0
    data["w"][:, 24] = 0
    scalers = fit_scalers(data)
    data["x"][0, 0, 6] = 1
    data["w"][0, 24] = 1
    transformed = scale(data, scalers)
    assert transformed["x"][0, 0, 6] == 1
    assert transformed["w"][0, 24] == 1


def test_embedding_deepsets_cannot_observe_future_responses():
    data = batch()
    torch.manual_seed(8)
    data["unique_h"] = torch.randn(2, 4, 3)
    data["unique_x"] = data["x"].clone()
    data["response_owner"] = torch.arange(2)
    model = PGM(4, mode="deepsets", embedding_dim=3)
    before = model(data)[1].detach()
    # Second example has only one observed response. Its other embeddings and
    # individual features exist in the offline bank but must have no effect.
    data["unique_h"][1, 1:] += 100
    data["unique_x"][1, 1:] -= 100
    assert torch.allclose(before, model(data)[1], atol=1e-6)
