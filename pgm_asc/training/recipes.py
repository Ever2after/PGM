"""Shared model-arm definitions used by training and evaluation."""
from __future__ import annotations


def recipes():
    result=[{'selector':s,'kind':k,'seed':42} for s in ['M','S'] for k in ['mlp','unmasked','masked']]
    result += [{'selector':'S','kind':k,'seed':seed} for seed in [43,44] for k in ['mlp','unmasked','masked']]
    return result


def name(recipe):return f"{recipe['selector']}_{recipe['kind']}_seed{recipe['seed']}"
