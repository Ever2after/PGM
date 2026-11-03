"""Public command-line interface for the integrated experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from pgm_asc.core.config import CONFIG, PROJECT, RUN, stage_config
from pgm_asc.core.io import read_json


def main():
    parser = argparse.ArgumentParser(description='Integrated PGM adaptive self-consistency experiment')
    parser.add_argument('command', choices=['status', 'validate', 'train', 'collect', 'calibrate', 'evaluate', 'run', 'report'])
    parser.add_argument('--stage', choices=['selector', 'state', 'all'], default='all')
    parser.add_argument('--config', default=str(CONFIG))
    parser.add_argument('--split', choices=['calibration', 'test'], default='calibration')
    parser.add_argument('--replay', action='store_true', help='Recompute evaluation from the retained bank')
    args = parser.parse_args()
    config_path = Path(args.config).resolve(); os.chdir(PROJECT)
    stages = ['selector', 'state'] if args.stage == 'all' else [args.stage]
    if args.command == 'status':
        for stage in stages:
            root = RUN / stage
            value = read_json(root / 'evaluation.json') if (root / 'evaluation.json').exists() else {}
            print(stage, 'complete' if value else 'pending', 'questions:', value.get('questions', 0))
        return
    if args.command == 'validate':
        from pgm_asc.experiments.validation import validate
        print(validate(stages)); return
    if args.command == 'report':
        from pgm_asc.evaluation.report import report
        print(report()); return
    import pgm_asc.experiments.pipeline as pipeline
    if args.command == 'run':
        pipeline.run(args.stage, config_path); return
    for stage in stages:
        config = stage_config(stage, config_path)
        if args.command == 'train':
            if stage == 'selector':
                pipeline.train_selector(config)
            else:
                from pgm_asc.training.state import run as train
                from pgm_asc.training.value import run as values
                train(config); values(config)
        elif args.command == 'collect':
            import pgm_asc.collection.banks as collect
            (collect.calibration if args.split == 'calibration' else collect.holdout)(config)
        elif args.command == 'calibrate':
            if stage == 'state':
                pipeline.model_lock(config)
            pipeline.calibrate(stage, config)
        elif args.command == 'evaluate':
            result = pipeline.evaluate(stage, config, replay=args.replay)
            print(stage, 'evaluated questions:', result['questions'])


if __name__ == '__main__':
    main()
