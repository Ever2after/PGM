"""Observed-only continuation features and frozen value prediction."""
from __future__ import annotations

import numpy as np


def continuation_features(histories, data):
    """Identical observed-only feature family for PGM and non-PGM controls."""
    n = len(histories[0]["chosen"])
    rows = []
    for qi, history in enumerate(histories):
        for index, chosen in enumerate(history["chosen"]):
            prefix = np.asarray(history["chosen"][:index + 1])
            run = 0
            for value in prefix[::-1]:
                if value != chosen:
                    break
                run += 1
            rows.append([np.log1p(index + 1), history["confidence"][index],
                         history.get("other", np.zeros(n))[index],
                         float(history["can_stop"][index]), run / (index + 1),
                         float((prefix == chosen).mean()),
                         *data["w"][qi * n + index].numpy().tolist()])
    return np.asarray(rows, dtype=np.float32)


def continuation_predict(fitted, histories, data):
    values = fitted["model"].predict(continuation_features(histories, data)) * fitted["output_scale"]
    values[:, 1] = np.maximum(values[:, 1], 1.)
    return values.reshape(len(histories), -1, 2)
