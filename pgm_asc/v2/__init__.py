"""PGM v2: a small generative model of the sampled answers, fitted by maximum likelihood and replayed on the v1 banks.

Modules: ``data`` (compact per-bank arrays and response similarities), ``model`` (likelihood, posterior, fit),
``voi`` (Monte-Carlo value of one more sample), ``baselines`` (SC/ASC/ESC/ReASC threshold families),
``run`` (fit + sweep + select + compare for one cell), ``report`` (tables and figures).
"""
from __future__ import annotations

from pgm_asc.config import ARTIFACTS, RESULTS, ROOT
from pgm_asc.utils import read_json

CONFIG = read_json(ROOT / "configs" / "v2.json")
SLOTS = 16          # candidate slots per question (at most 16 distinct answers among 16 responses)
OTHER = SLOTS       # hypothesis index: the correct answer has not been observed


def data_path(model: str, dataset: str):
    return ARTIFACTS / "v2" / f"{model}_{dataset}.npz"


def result_path(model: str, dataset: str):
    return RESULTS / "v2" / "cells" / f"{model}_{dataset}.json"
