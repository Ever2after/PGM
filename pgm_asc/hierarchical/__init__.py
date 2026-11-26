"""Question-level random-effect CRP structure ablation."""
from pgm_asc.config import ROOT
from pgm_asc.utils import read_json
from pgm_asc.objectives import MODEL_CONFIG

CONFIG = read_json(ROOT / "configs" / "hierarchical.json")
