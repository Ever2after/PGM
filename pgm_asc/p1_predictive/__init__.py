"""P1 mixed objective plus label-free next-answer predictive log score."""
from pgm_asc.config import ROOT
from pgm_asc.utils import read_json

CONFIG=read_json(ROOT/'configs'/'p1_predictive.json')
