"""Shared prefix buckets and training-objective weights."""
from __future__ import annotations

import numpy as np
import torch


def training_weights():
    return torch.tensor([0.,1/3,1/6,1/6,1/12,1/12,1/12,1/12])


BUCKET_ENDS = np.array([1, 2, 4, 8, 16, 32])


def bucket_ids(times):
    times = np.asarray(times)
    if np.any((times < 1) | (times > 32)):
        raise ValueError("Prefixes must be in [1,32]")
    return np.searchsorted(BUCKET_ENDS, times, side="left")


def prefix_weights(times):
    buckets = bucket_ids(times)
    counts = np.bincount(buckets, minlength=6)
    weights = 1. / counts[buckets]
    return weights / weights.sum()
