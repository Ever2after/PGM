import pytest
from pgm_asc.core.io import write_json
from pgm_asc.collection import study
from tests.unit.test_reasc_fast import Engine,Tokenizer,config


def test_generic_collector_gsm_parser_and_2048_budget(monkeypatch,tmp_path):
    cfg=config(tmp_path,split='dev',n=2)|{'dataset':'gsm8k','max_new_tokens':2048,'repetition_penalty':1.}
    # The collector's shared parity receipt is separate from its output root.
    # Keep it inside the test sandbox rather than relying on a completed run.
    monkeypatch.chdir(tmp_path)
    write_json('artifacts/runs/reasc_pgm/reference_validation.json', {'passed': True})
    e=Engine()
    # Fake default includes repetition penalty, unlike older math-only mock.
    original=e.get_default_sampling_params
    def params():
        p=original();p.repetition_penalty=1.;return p
    e.get_default_sampling_params=params
    monkeypatch.setattr(study,'initialize',lambda cfg:(e,Tokenizer()))
    monkeypatch.setattr(study.importlib.metadata,'version',lambda name:'fake')
    study.collect(cfg,'dev',0,1)
    assert e.calls[0][1][0].max_tokens==2048
    assert 'numerical integer' in e.calls[1][0][0]


def test_all_eval_split_names_require_policy_lock(tmp_path):
    cfg=config(tmp_path,split='confirm',n=1)|{'dataset':'math','max_new_tokens':1024}
    with pytest.raises(FileNotFoundError):study.collect(cfg,'confirm',0,1)
