"""Direct multi-step repair/damage training for fixed P1 graphs."""
from pgm_asc.config import ROOT
from pgm_asc.utils import read_json

CONFIG=read_json(ROOT/'configs'/'p1_transitions.json')
