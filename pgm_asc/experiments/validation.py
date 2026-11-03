"""Retained-cohort, decision, symlink, and artifact validation."""
from __future__ import annotations

import os

from pgm_asc.core.config import DATA, RUN
from pgm_asc.core.integrity import problem_key, verify_lock
from pgm_asc.core.io import read_json, read_jsonl, utc_now, write_json


def _links(root):
    stack = [str(root)]; broken = []; count = 0
    while stack:
        folder = stack.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_symlink():
                    count += 1
                    if not os.path.exists(entry.path):
                        broken.append(entry.path)
                elif entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
    if broken:
        raise ValueError(f'Broken retained symlinks: {broken[:5]}')
    return count


def validate(stages=('selector', 'state')):
    result = {'created_at': utc_now(), 'stages': {}}
    for stage in stages:
        root = RUN / stage
        lock = read_json(root / 'selection_lock.json'); verify_lock(lock)
        evaluation = read_json(root / 'evaluation.json')
        count = 0
        for p in (root / 'decisions').glob('*.jsonl'):
            rows = read_jsonl(p)
            if len(rows) != evaluation['questions']:
                raise ValueError(f'Decision count differs from holdout: {p}')
            if len({r['question_id'] for r in rows}) != len(rows):
                raise ValueError(f'Duplicate question decisions: {p}')
            if stage == 'state' and any(not 2 <= r['t'] <= 8 for r in rows):
                raise ValueError(f'State policy violates minimum/cap: {p}')
            count += 1
        result['stages'][stage] = {'questions': evaluation['questions'], 'locked_files': len(lock['artifact_sha256']),
                                   'decision_files': count, 'symlinks_verified': _links(root)}
    # Packed training examples preserve the original question-level units.
    train = read_jsonl(DATA / 'state_train/train.jsonl'); dev = read_jsonl(DATA / 'state_train/dev.jsonl')
    support = read_jsonl(DATA / 'selector_train/train.jsonl')
    keys = [{problem_key(q['problem']) for q in qs} for qs in [train, dev, support]]
    if len(train) != 900 or len(dev) != 700 or len(support) != 800 or keys[0] & (keys[1] | keys[2]):
        raise ValueError('Training cohort roles or overlap changed')
    result['training_questions'] = {'selector': 800, 'selector_dev': 200, 'state': 900, 'state_dev': 700}
    result['passed'] = True
    write_json(RUN / 'validation.json', result)
    return result
