import numpy as np
import torch

from pgm_asc.collection.confidence import bottom_group
from pgm_asc.collection.confidence import token_certainty


def test_full_vocabulary_certainty_matches_log_softmax():
    logits = torch.tensor([[1., 2., -3.], [0., 0., 0.]])
    assert torch.allclose(token_certainty(logits), -logits.log_softmax(-1).mean(-1))
    assert torch.allclose(token_certainty(logits)[1], torch.tensor(np.log(3), dtype=torch.float32))


def test_bottom_group_matches_direct_windows_and_short_response():
    values = np.arange(150, dtype=float)
    groups = [np.mean(values[i:i+128]) for i in range(len(values)-127)]
    expected = np.mean(sorted(groups)[:int(np.ceil(len(groups)*.1))])
    assert abs(bottom_group(values)-expected) < 1e-9
    assert bottom_group([1., 3.]) == 2.
    assert bottom_group([]) == 0.
