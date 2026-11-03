from pathlib import Path
import ast
import numpy as np
import torch
import pytest
from pgm_asc.core.io import read_json
from pgm_asc.core.integrity import load_tensor
from pgm_asc.core.config import stage_config
from pgm_asc.core.config import runtime_config


def test_unified_config_resolves_independent_stage_sizes():
    selector = stage_config('selector'); state = stage_config('state')
    assert selector['samples_per_question'] == 32
    assert state['samples_per_question'] == 8
    assert selector['followup_sizes']['test'] == 500
    assert state['followup_sizes']['test'] == 1000
    assert selector['model_revision'] == state['model_revision']
    assert runtime_config('configs/experiment.json') == state


def test_package_imports_follow_the_nested_layout():
    import importlib.util

    package = Path('pgm_asc')
    layers = {'core', 'data', 'models', 'training', 'policies', 'collection', 'evaluation', 'experiments'}
    assert {p.stem for p in package.glob('*.py')} == {'__init__', '__main__', 'cli'}
    assert {p.name for p in package.iterdir() if p.is_dir() and p.name != '__pycache__'} == layers
    lower_layers = {'core', 'models', 'data', 'policies'}
    forbidden = {'training', 'experiments', 'evaluation', 'collection'}
    for p in package.rglob('*.py'):
        source = p.read_text(); tree = ast.parse(source)
        assert 'pgm_pilot' not in source
        assert 'gemma3_pgm_v' not in source
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ''
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('pgm_asc.'):
                        assert importlib.util.find_spec(alias.name), (p, alias.name)
                continue
            else:
                continue
            if module.startswith('pgm_asc.'):
                assert importlib.util.find_spec(module), (p, module)
                if p.parts[1] in lower_layers:
                    assert module.split('.')[1] not in forbidden, (p, module)


def test_source_fingerprints_include_nested_modules():
    from pgm_asc.core.integrity import source_files

    files = source_files()
    assert Path('pgm_asc/models/state.py') in files
    assert Path('pgm_asc/collection/vllm.py') in files
    assert Path('pgm_asc/experiments/selector.py') in files
    assert set(files) == set(Path('pgm_asc').rglob('*.py'))


def test_collection_subprocesses_use_new_modules(tmp_path, monkeypatch):
    from pgm_asc.collection import banks

    config = {'run_dir': str(tmp_path / 'run'), 'data_dir': str(tmp_path / 'data'), 'samples_per_question': 8}
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(banks.subprocess, 'run', lambda args, **kwargs: calls.append((args, kwargs)))
    audited = []
    monkeypatch.setattr(banks, 'audit_bank', lambda cfg, split: audited.append(split))
    banks.collect_one(config, 'calibration_a', 4)
    assert [args[3] for args, _ in calls] == [
        'pgm_asc.collection.vllm',
        'pgm_asc.collection.teacher_forcing',
        'pgm_asc.collection.confidence',
    ]
    assert Path(calls[0][0][0]).parents[1].name == '.venv-vllm'
    assert all(Path(args[0]).parents[1].name == '.venv' for args, _ in calls[1:])
    assert all(kwargs['env']['CUDA_VISIBLE_DEVICES'] == '4' for _, kwargs in calls)
    assert audited == ['calibration_a']


def test_restricted_tensor_loader_supports_numeric_scalers(tmp_path):
    path = tmp_path / 'state.pt'
    expected = {'state_dict': {'weight': torch.arange(4.)},
                'scaler': {'mean': np.array([1., 2.]), 'std': np.array([.1, .2], dtype=np.float32)},
                'train_indices': [0, 1]}
    torch.save(expected, path)
    actual = load_tensor(path)
    assert torch.equal(actual['state_dict']['weight'], expected['state_dict']['weight'])
    assert np.array_equal(actual['scaler']['mean'], expected['scaler']['mean'])
    assert np.array_equal(actual['scaler']['std'], expected['scaler']['std'])


def test_empty_bank_is_rejected(tmp_path):
    from pgm_asc.collection.banks import audit_bank
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/test.jsonl').write_text('')
    with pytest.raises(ValueError, match='membership'):
        audit_bank({'data_dir': str(tmp_path / 'data'), 'run_dir': str(tmp_path / 'run'), 'samples_per_question': 8}, 'test')
