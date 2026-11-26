"""P1 Mixed plus exact multi-step repair/damage Brier loss."""
from pgm_asc.config import ROOT
from pgm_asc.utils import read_json

CONFIG=read_json(ROOT/'configs'/'p1_brier.json')
