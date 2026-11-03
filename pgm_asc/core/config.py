"""Project paths and resolved configuration for the two experimental stages."""
from __future__ import annotations

from pathlib import Path

from pgm_asc.core.io import read_json


PROJECT = Path(__file__).resolve().parents[2]
RUN = Path('artifacts/runs/pgm_asc')
DATA = Path('artifacts/data/pgm_asc')
BOOTSTRAP = RUN / 'bootstrap'
SELECTOR = RUN / 'selector'
STATE = RUN / 'state'
CONFIG = Path('configs/experiment.json')


def stage_config(stage, path=CONFIG):
    if stage not in ('selector', 'state'):
        raise ValueError(f'Unknown stage: {stage}')
    value = read_json(path)
    config = {**value['common'], **value[stage]}
    config.update(experiment=f"{value['experiment']}_{stage}",
                  run_dir=str(RUN / stage), data_dir=str(DATA / stage),
                  source_run_dir=str(BOOTSTRAP), source_data_dir=str(DATA / 'selector_train'),
                  config_path=str(path), primary_prefixes=[8, 16, 32] if stage == 'selector' else [2, 4, 8])
    return config


def runtime_config(path, stage='state'):
    value = read_json(path)
    return stage_config(stage, path) if 'common' in value else value
