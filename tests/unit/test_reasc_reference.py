import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from pgm_asc.policies.reasc_reference import (
    clean, confidence_groups, decide, exact_groups, first_box,
    offline_calibrate, online_calibrate, prediction, trace,
)


def reference_clean():
    """Execute only the two audited pure parsing definitions, never notebook cells."""
    notebook = json.loads(Path('third_party/reasc_official/confsc_offline_code.ipynb').read_text())
    nodes = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            module = ast.parse(''.join(cell['source']))
        except SyntaxError:
            continue
        nodes.extend(n for n in module.body if isinstance(n, ast.FunctionDef)
                     and n.name in ['clean', 'extract_boxed_content'])
    assert len(nodes) == 2
    ns = {'dataset': 'math', 're': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<audited reference parsers>', 'exec'), ns)
    return ns


@pytest.mark.parametrize('text', [r'a \boxed{\frac{1}{2}} b \boxed{3}',
                               r'\boxed{ a {b}', 'unboxed', r'\boxed{}'])
def test_parser_official_parity(text):
    ref = reference_clean()
    assert first_box(text) == ref['extract_boxed_content'](text)
    assert clean(text) == ref['clean'](text)


@pytest.mark.parametrize('length', [1, 9, 10, 19, 127, 128, 129, 1024])
def test_confidence_includes_short_suffixes_and_floor(length):
    v = np.random.default_rng(length).normal(size=length)
    windows = [np.mean(v[i:i+128]) for i in range(length)]
    expected = sorted(windows)[:max(1, length // 10)]
    result = confidence_groups(v)
    assert result['lowest'] == pytest.approx(min(windows), abs=1e-12)
    assert result['bottom10'] == pytest.approx(np.mean(expected), abs=1e-12)


def test_exact_strings_empty_and_tie_order():
    reps, ids = exact_groups(['1/2', r'\frac{1}{2}', '', '1/2', ''])
    assert reps == ['1/2', r'\frac{1}{2}', '']
    assert ids.tolist() == [0, 1, 2, 0, 2]
    assert prediction('no box', r'\boxed{ 7 }') == '7'
    assert decide(['b', 'a'] * 8, [0] * 16, 'sc16')[1] == 0


def test_beta_rule_and_weighted_first_stop_without_gate():
    h = trace(['a', 'b', 'a', 'a'], [0]*4, 'asc')
    expected = quad(lambda p: p**3 * (1-p), .5, 1)[0] / quad(lambda p: p**3 * (1-p), 0, 1)[0]
    assert h['confidence'][-1] == pytest.approx(expected)
    cfg = {'mean': 0., 'std': 1., 'gate': 100.}
    assert decide(['a']*16, [3.]*16, 'reasc_offline', cfg)[0] == 0


def test_esc_uses_accumulated_prefix_majority():
    predictions = ['a','a','a','c', 'a','a','a','c', 'b','b','b','b', 'd','d','d','d']
    assert decide(predictions, [0]*16, 'esc')[:2] == (11, 0)


def test_calibration_guards_and_label_free_online():
    a = offline_calibrate([0,1,2,3], [False, True, False, True])
    assert a['mean'] == 1.5 and a['gate'] == 3.
    assert a['fit_score'] == 'lowest_group' and a['inference_score'] == 'bottom10'
    b = offline_calibrate([1,1], [False,False])
    assert b['zero_std_guard'] and b['no_correct_guard']
    c = online_calibrate(np.r_[np.linspace(0,1,32),np.linspace(5,6,32)])
    assert c['uses_labels'] is False and 1 < c['gate'] < 7


def test_future_does_not_change_observed_decision():
    a = trace(['a']*4 + ['b']*12, [0]*16, 'asc')
    b = trace(['a']*4 + ['c','d']*6, [99]*16, 'asc')
    np.testing.assert_array_equal(a['chosen'][:4], b['chosen'][:4])
    np.testing.assert_allclose(a['confidence'][:4], b['confidence'][:4])
