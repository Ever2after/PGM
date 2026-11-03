import pytest
from pgm_asc.core.io import read_json
from pgm_asc.experiments.qwen7_only import cached_loader


def test_loader_reuses_generator_across_datasets_without_reload():
    calls=[];obj=object()
    def factory(path,**kwargs):calls.append(path);return obj
    load=cached_loader(factory)
    assert load('same-checkpoint',torch_dtype='bf16') is obj
    assert load('same-checkpoint',torch_dtype='bf16') is obj
    assert calls==['same-checkpoint']
    with pytest.raises(ValueError):load('different-checkpoint')


def test_scope_defers_all_omni_and_keeps_original_protocol():
    scope=read_json('configs/pgm_followup_runtime_scope.json')
    assert len(scope['active_jobs'])==2 and all('qwen7b' in p and 'omnimath' not in p for p in scope['active_jobs'])
    assert len(scope['deferred_jobs'])==3 and all('omnimath' in p for p in scope['deferred_jobs'])
    assert len(read_json('configs/pgm_followup.json')['jobs'])==9
