"""End to end on the synthetic study: fit every head, save the policy, evaluate, and replay deterministically."""
import json


def test_fit_and_evaluate(study):
    from pgm_asc.config import ROOT, Cell, decisions_path, evaluation_path, policy_path
    from pgm_asc.data.bank import load_bank
    from pgm_asc.evaluate import evaluate_cell
    from pgm_asc.fit import fit_cell

    banks = load_bank("qwen3b", "math", "test")
    assert len(banks) == 8 and banks[0].vectors.shape == (16, 8)
    texts = load_bank("qwen3b", "math", "test", columns=["text"], embeddings=False)
    assert texts[0].records[0]["text"] and texts[0].vectors is None
    cell = Cell("qwen3b", "math")
    for path in (evaluation_path(cell.name), policy_path(cell.policy_name), decisions_path(cell.name)):
        assert not path.is_relative_to(ROOT), f"test would write into the repository: {path}"
    policy = fit_cell(cell)
    assert len(policy["heads"]) == 2 * 12
    assert policy_path(cell.policy_name).exists()
    first = evaluate_cell(cell)
    assert decisions_path(cell.name).exists()
    for name in ("sc16", "asc", "reasc_offline", "mixture_s42__selected", "deepsets_embedding_s43__fixed95"):
        assert 0 <= first["summary"][name]["correct"] <= 1
    assert set(first["primary_structure_contrasts"]) == {"alpha1", "random", "candidate_confidence", "deepsets_embedding"}
    second = evaluate_cell(cell)
    assert second["summary"] == first["summary"]
    assert json.loads(evaluation_path(cell.name).read_text())["cell"] == cell.name
