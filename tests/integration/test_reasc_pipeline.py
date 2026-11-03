"""Small synthetic end-to-end fit/lock/replay; no benchmark outcomes read."""
import copy
from pathlib import Path
import numpy as np
import torch
from pgm_asc.core.io import read_json, write_json
from pgm_asc.core.integrity import verify_lock, file_sha
from pgm_asc.models.selector import PGM
from tests.unit.test_reasc_adapter import bank,rep


def test_select_lock_and_evaluate_synthetic(monkeypatch,tmp_path):
    import pgm_asc.experiments.reasc_comparison as training
    import pgm_asc.evaluation.reasc_comparison as evaluation
    from pgm_asc.data.reasc_bank import build
    from pgm_asc.data.banks import fit_scalers
    # Exercise the historical transfer boundary with synthetic locked inputs.
    # Never read a researcher's real checkpoint or start a second environment.
    monkeypatch.chdir(tmp_path)
    oldroot = Path('artifacts/runs/pgm_asc/selector')
    checkpoint = oldroot / 'refits/mixture/checkpoint.pt'
    checkpoint.parent.mkdir(parents=True)
    model = PGM(32)
    torch.save({'state_dict': model.state_dict(), 'spec': {'mode': 'mixture'}, 'n': 32}, checkpoint)
    head = oldroot / 'synthetic_continuation.json'
    write_json(head, {'fixture': 'zero gain, unit cost'})
    old = {'calibrator': {'kind': 'global', 'T': 1.},
           'policy': {'minimum': 2, 'patience': 1, 'cap': 8, 'gamma': .95, 'continuation_lambda': 0.},
           'continuation_head': str(head)}
    write_json(oldroot / 'selection_lock.json', {
        'stages': {'s6_continuation': {'mixture': {'selected': old}}},
        'artifact_sha256': {str(p): file_sha(p) for p in [checkpoint, head]},
    })
    transfer_calls = []
    def transfer_boundary(args, **kwargs):
        transfer_calls.append(args)
        features = np.load(args[-2], allow_pickle=False)
        assert features.shape == (64, 31)
        values = np.column_stack([np.zeros(len(features)), np.ones(len(features))])
        np.save(args[-1], values, allow_pickle=False)
    monkeypatch.setattr(evaluation.subprocess, 'run', transfer_boundary)
    def banks(cfg,split):
        output=[]
        for i in range(4):
            b=bank(f'{split}_{i}');b['labels'].update(raw_response_correct=[True]*16,elapsed_seconds=.1,input_tokens=100,output_tokens=20)
            for r in b['records']:
                r.update(lowest_confidence=float(i),bottom10_confidence=float(i),feature_input_tokens=30,
                         feature_seconds_share=.1,native_generation_seconds_share=.2,
                         extraction_inline={'input_tokens':20,'output_tokens':3,'seconds_share':.1},
                         extraction_final={'input_tokens':20,'output_tokens':3,'seconds_share':.1})
            output.append(b)
        return output
    data,_=build(banks(None,'train'),rep());scalers=fit_scalers(data)
    monkeypatch.setattr(training,'load',banks);monkeypatch.setattr(evaluation,'load',banks)
    monkeypatch.setattr(training,'frozen_preprocessing',lambda:(rep(),scalers))
    cfg={'run_dir':str(tmp_path),'data_dir':str(tmp_path/'cohorts'),'label':'synthetic validation',
         'caps':[8,16],'lambdas':[0.], 'train':{'seed':42,'lr':.001,'weight_decay':.001,'question_batch':4,'epochs':2,'patience':20}}
    training.select(cfg)
    lock=read_json(tmp_path/'selection_lock.json');verify_lock(lock)
    assert not lock['holdout_accessed'] and len(lock['methods'])==6
    result=evaluation.evaluate(cfg,'synthetic_holdout')
    assert len(result['summary'])==12
    assert result['summary']['pgm_mixture']['questions']==4
    assert (tmp_path/'synthetic_holdout/report.md').exists()
    assert len(transfer_calls) == 1
