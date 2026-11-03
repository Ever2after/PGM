"""Selector-to-state orchestration and model/holdout barriers."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import torch
from filelock import FileLock

from pgm_asc.core.config import CONFIG, DATA, RUN, SELECTOR, STATE, stage_config
from pgm_asc.core.integrity import file_sha, load_tensor, problem_key, source_files, verify_lock
from pgm_asc.core.io import digest, log, read_json, read_jsonl, utc_now, write_json


def train_selector(config):
    torch.set_num_threads(2)
    from pgm_asc.experiments.selector_context import Context
    from pgm_asc.training.selector import fit
    from pgm_asc.training.selector import train_bucket_model
    ctx = Context(config)
    train, _ = ctx.data('train'); dev, _ = ctx.data('dev')
    for method, choice in ctx.choices.items():
        initial = Path(choice['path'])
        if not (initial / 'checkpoint.pt').exists():
            fit(train, dev, choice['spec'], 32, initial, seed=42,
                lr=choice['lr'], weight_decay=choice['weight_decay'], epochs=config['followup_epochs'])
        if method in config['refit_methods']:
            train_bucket_model(train, dev, choice['spec'], SELECTOR / 'refits' / method,
                               choice['lr'], choice['weight_decay'],
                               epochs=config['followup_epochs'], device='cpu')


def preflight_state(config):
    from pgm_asc.data.state import load
    from pgm_asc.data.state import targets
    from pgm_asc.data.state import observed_features
    from pgm_asc.data.state import selector_outputs
    from pgm_asc.models.state import transitions
    from pgm_asc.models.state import feasible
    from pgm_asc.experiments.state_context import Context
    train, _ = load('train'); dev, _ = load('dev')
    questions = {s: read_jsonl(DATA / 'state_train' / f'{s}.jsonl') for s in ['train', 'dev']}
    train_keys = {problem_key(q['problem']) for q in questions['train']}
    dev_keys = {problem_key(q['problem']) for q in questions['dev']}
    support = {problem_key(q['problem']) for q in read_jsonl(DATA / 'selector_train/train.jsonl')}
    if len(train_keys) != config['state_train_questions'] or len(dev_keys) != config['state_dev_questions']:
        raise ValueError('State training/development cohort count changed')
    if train_keys & (dev_keys | support):
        raise ValueError('State training overlap')
    for split, data in [('train', train), ('dev', dev)]:
        outputs = selector_outputs(data, split)
        for selector in ['M', 'S']:
            obs = observed_features(data, outputs[selector]); y, _ = targets(data, obs)
            allowed = transitions(torch.from_numpy(obs['new']), torch.from_numpy(obs['changed'])).numpy()
            for t in range(1, 8):
                if not allowed[np.arange(len(y)), t, y[:, t-1], y[:, t]].all():
                    raise ValueError('True state path violates transition mask')
            if not feasible(torch.from_numpy(obs['count'])).numpy()[np.arange(len(y))[:, None], np.arange(8)[None, :], y].all():
                raise ValueError('True state violates support mask')
    reference = Context(config).frozen('dev')['rows'][200:]
    original = read_jsonl(SELECTOR / 'decisions/s6_continuation_mixture.jsonl')
    if len(reference) != len(original) or any(
        any(a[k] != b[k] for k in ['question_id', 't', 'chosen', 'correct', 'token_cost'])
        for a, b in zip(reference, original)):
        raise ValueError('Frozen selector decisions changed on transferred holdout')
    from pgm_asc.training.recipes import recipes
    from pgm_asc.training.recipes import name
    audited_folds = 0
    for recipe in recipes():
        directory = STATE / 'models' / name(recipe)
        bundle = joblib.load(directory / 'policy_bundle.joblib')
        with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
            folds = saved['folds'].copy()
        for fold in range(5):
            model = load_tensor(directory / f'fold{fold}.pt')
            expected = np.flatnonzero(folds != fold).tolist()
            if model['train_indices'] != expected or bundle['folds'][fold]['train_indices'] != expected:
                raise ValueError(f'State/value outer-fold training membership changed: {name(recipe)} fold{fold}')
            if set(expected).intersection(np.flatnonzero(folds == fold)):
                raise ValueError('Outer-fold label overlap')
            audited_folds += 1
    result = {'created_at': utc_now(), 'state_training_questions': len(train_keys),
              'development_questions': len(dev_keys), 'support_train_overlap': 0, 'train_dev_overlap': 0,
              'exact_frozen_selector_replay_questions': len(reference), 'logical_state_masks': 'passed',
              'historical_selector_selection_not_OOF': True, 'new_state_value_outer_folds_audited': audited_folds}
    write_json(STATE / 'validation.json', result)
    return result


def model_lock(config):
    path = STATE / 'model_lock.json'
    if path.exists():
        result = read_json(path); verify_lock(result); return result
    preflight_state(config)
    source = read_json(SELECTOR / 'selection_lock.json'); verify_lock(source)
    files = [CONFIG, Path('docs/METHOD.md'), SELECTOR / 'selection_lock.json', STATE / 'training_results.json']
    files += source_files()
    files += list((STATE / 'models').glob('*/*.pt')) + list((STATE / 'models').glob('*/policy_bundle.joblib'))
    files += list((STATE / 'models').glob('*/oof_policy_audit.json'))
    files += list((STATE / 'cache/train').glob('*')) + list((STATE / 'cache/dev').glob('*'))
    hashes = dict(source['artifact_sha256'])
    hashes.update({str(p): file_sha(p) for p in files if p.is_file()})
    result = {'created_at': utc_now(), 'primary': config['primary_model'], 'artifact_sha256': hashes,
              'fresh_calibration_accessed': False, 'holdout_accessed': False,
              'state_training_questions': config['state_train_questions'], 'development_questions': config['state_dev_questions']}
    result['protocol_hash'] = digest(result); write_json(path, result)
    return result


def calibrate(stage, config):
    if stage == 'selector':
        from pgm_asc.experiments.selector import select_stages
        return select_stages(config)
    from pgm_asc.experiments.state import lock_calibration
    return lock_calibration(config)


def evaluate(stage, config, replay=False):
    torch.set_num_threads(2)
    root = Path(config['run_dir'])
    verify_lock(read_json(root / 'selection_lock.json'))
    if not replay and (root / 'evaluation.json').exists():
        return read_json(root / 'evaluation.json')
    if stage == 'selector':
        from pgm_asc.evaluation.selector import evaluate_holdout
        return evaluate_holdout(config, force=replay)
    from pgm_asc.evaluation.state import evaluate as evaluate_state
    return evaluate_state(config)


def run(stage='all', config_path=CONFIG):
    torch.set_num_threads(2)
    import pgm_asc.collection.banks as collect
    stages = ['selector', 'state'] if stage == 'all' else [stage]
    RUN.mkdir(parents=True, exist_ok=True)
    with FileLock(str(RUN / '.pipeline.lock'), timeout=0):
        for name in stages:
            config = stage_config(name, config_path); root = Path(config['run_dir'])
            status = {'state': 'running', 'stage': name, 'pid': os.getpid(), 'started_at': utc_now()}
            def phase(label):
                status.update(phase=label, updated_at=utc_now()); write_json(root / 'status.json', status)
                log('pipeline phase', stage=name, phase=label)
            try:
                if (root / 'evaluation.json').exists():
                    phase('verify_completed_stage'); result = evaluate(name, config)
                else:
                    phase('training')
                    if name == 'selector':
                        train_selector(config)
                    else:
                        from pgm_asc.training.state import run as train
                        from pgm_asc.training.value import run as values
                        verify_lock(read_json(SELECTOR / 'selection_lock.json'))
                        train(config); values(config); model_lock(config)
                    phase('calibration_banks'); collect.calibration(config)
                    phase('calibration_and_policy_lock'); calibrate(name, config)
                    phase('locked_holdout_bank'); collect.holdout(config)
                    phase('evaluation'); result = evaluate(name, config)
                status.update(state='complete', questions=result['questions']); phase('complete')
            except BaseException as error:
                status.update(state='failed', error=str(error)); phase('failed'); raise
