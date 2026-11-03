"""Pinned ReASC notebook semantics, with label-free policy inference."""
from __future__ import annotations

import re
import numpy as np
from scipy.special import betainc
from sklearn.mixture import GaussianMixture


def first_box(text):
    start = text.find(r'\boxed{')
    if start < 0:
        return None
    i = start + len(r'\boxed{'); depth = 1; body = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0: break
        body.append(c); i += 1
    return ''.join(body)


def clean(text, dataset='math'):
    if dataset == 'gsm8k':
        numbers = re.findall(r'\d+(?:\.\d+)?', text.replace(',', ''))
        return numbers[-1] if numbers else None
    value = first_box(text)
    if value is None:
        if dataset in ['gpqa_diamond', 'arcChallenge']:
            letters = re.findall(r'A|B|C|D|a|b|c|d', text)
            return letters[0].lower() if letters else ''
        return ''
    value = value.replace(' ', '')
    return value.lower() if dataset in ['gpqa_diamond', 'arcChallenge'] else value


def prediction(response, extracted, dataset='math'):
    if dataset == 'gsm8k':
        return clean(extracted.replace(' ', '').strip(), dataset)
    value = clean(response, dataset)
    if value == '': value = clean(extracted, dataset)
    if value == '': value = extracted.replace(' ', '').strip()
    return value


def exact_groups(predictions):
    reps = []; ids = []
    for value in predictions:
        if value not in reps: reps.append(value)
        ids.append(reps.index(value))
    return reps, np.asarray(ids, dtype=np.int64)


def confidence_groups(token_scores, window=128):
    values = np.asarray(token_scores, dtype=np.float64)
    if not len(values):
        return {'lowest': 0., 'bottom10': 0., 'empty': True}
    c = np.r_[0., np.cumsum(values)]; start = np.arange(len(values)); end = np.minimum(start + window, len(values))
    averages = (c[end] - c[start]) / (end - start)
    count = max(1, len(averages) // 10)
    return {'lowest': float(averages.min()), 'bottom10': float(np.partition(averages, count - 1)[:count].mean()), 'empty': False}


def offline_calibrate(confidences, correct, target=.9):
    scores = np.asarray(confidences, dtype=np.float64); correct = np.asarray(correct, dtype=bool)
    best = None; best_acc = None; reached = False
    for threshold in np.sort(scores):
        mask = scores >= threshold; acc = float(correct[mask].mean())
        if acc >= target:
            best, best_acc, reached = float(threshold), acc, True; break
        if best_acc is None or acc > best_acc: best, best_acc = float(threshold), acc
    mean_correct = float(scores[correct].mean()) if correct.any() else None
    gate = max(best, mean_correct) if mean_correct is not None else float(scores.max())
    std = float(scores.std())
    return {'mean': float(scores.mean()), 'std': std if std > 0 else 1., 'zero_std_guard': std == 0,
            'gate': gate, 'target': target, 'target_reached': reached, 'no_correct_guard': not correct.any(),
            'lambda': .7, 'calibration_questions': len(scores), 'fit_score': 'lowest_group', 'inference_score': 'bottom10'}


def online_calibrate(first_scores, target=.9):
    scores = np.asarray(first_scores, dtype=np.float64)
    gmm = GaussianMixture(2, covariance_type='full', random_state=0).fit(scores[:, None])
    high = int(np.argmax(gmm.means_.ravel())); ordered = np.sort(scores)
    posterior = gmm.predict_proba(ordered[:, None])[:, high]; accepted = ordered[posterior >= target]
    tau = float(accepted[0]) if len(accepted) else float(ordered[-1]); std = float(scores.std())
    return {'mean': float(scores.mean()), 'std': std if std > 0 else 1., 'zero_std_guard': std == 0,
            'gate': max(float(gmm.means_[high, 0]), tau), 'target': target, 'lambda': .7,
            'calibration_questions': len(scores), 'uses_labels': False,
            'gmm': {'means': gmm.means_.ravel().tolist(), 'variances': gmm.covariances_.ravel().tolist(),
                    'weights': gmm.weights_.tolist(), 'converged': bool(gmm.converged_)}}


def trace(predictions, bottom_scores, method, calibration=None):
    _, ids = exact_groups(predictions); n = len(ids)
    weights = np.zeros(n); choices = []; confidences = []
    for i, label in enumerate(ids):
        value = 1.
        if method == 'reasc':
            z = (bottom_scores[i] - calibration['mean']) / calibration['std']
            # This is the notebook mapping. Reject nonfinite data instead of silently clipping evidence.
            value = max(1., float(np.exp(.7 * z)))
            if not np.isfinite(value): raise FloatingPointError('ReASC evidence overflow')
        weights[label] += value
        count = int(ids[:i+1].max()) + 1; chosen = int(weights[:count].argmax())
        ordered = sorted(weights[:count], reverse=True) + [0.]
        prob = float(1. - betainc(ordered[0] + 1, ordered[1] + 1, .5))
        if method == 'reasc' and i == 0 and bottom_scores[0] >= calibration['gate']: prob = 1.
        choices.append(chosen); confidences.append(prob)
    return {'chosen': np.asarray(choices), 'confidence': np.asarray(confidences), 'can_stop': np.ones(n, dtype=bool)}


def decide(predictions, bottom_scores, method, calibration=None):
    _, ids = exact_groups(predictions); n = len(ids)
    if method == 'pass1': return 0, int(ids[0]), 'fixed'
    if method == 'sc16': return n - 1, int(np.bincount(ids).argmax()), 'fixed'
    if method == 'esc':
        end = n
        for start in range(0, n, 4):
            block = ids[start:start+4]
            if len(set(block)) == 1: end = start + len(block); break
        return end - 1, int(np.bincount(ids[:end]).argmax()), 'unanimous_block' if end < n else 'cap'
    history = trace(predictions, bottom_scores, 'reasc' if method.startswith('reasc') else 'asc', calibration)
    hits = np.flatnonzero(history['confidence'] >= .95); index = int(hits[0]) if len(hits) else n - 1
    return index, int(history['chosen'][index]), 'confidence' if len(hits) else 'cap'
