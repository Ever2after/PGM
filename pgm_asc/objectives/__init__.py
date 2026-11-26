"""Controlled M1 training-objective ablation, separate from released studies."""

from pgm_asc.config import ROOT
from pgm_asc.utils import read_json
from pgm_asc.v2 import CONFIG as MODEL_CONFIG

CONFIG = read_json(ROOT / "configs" / "objectives.json")

# This study changes the loss only, including when resuming cached fits.
if MODEL_CONFIG["grid_size"] != 80 or MODEL_CONFIG["grid_logit_range"] != [-7.0, 7.0]:
    raise ValueError("Objective ablation keeps the released M1 grid: 80 nodes on logits [-7,7]")
