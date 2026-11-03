"""Artifact integrity and restricted numeric checkpoint loading."""
from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import torch

from pgm_asc.core.io import digest


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def problem_key(problem):
    return digest(' '.join(unicodedata.normalize('NFKC', problem).split()))


def load_tensor(path, *, mmap=False):
    # Saved state-head scalers contain NumPy arrays. Allow only their concrete
    # numeric constructors; arbitrary pickle globals remain disallowed.
    allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype,
               type(np.dtype('float32')), type(np.dtype('float64')), type(np.dtype('int64')),
               type(np.dtype('int32')), type(np.dtype('bool'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True, mmap=mmap)


def verify_lock(lock):
    for path, expected in lock['artifact_sha256'].items():
        if file_sha(path) != expected:
            raise ValueError(f'Artifact changed after lock: {path}')


def source_files():
    """Include every nested package module in experiment integrity locks."""
    return sorted(Path('pgm_asc').rglob('*.py'))
