from pathlib import Path
from pgm_asc.core.io import write_json,read_json
from pgm_asc.core.integrity import verify_lock
from tests.unit.test_reasc_adapter import bank


def fake_banks(cfg,split):
    result=[]
    for i in range(4):
        b=bank(f'{split}_{i}');b['labels'].update(raw_response_correct=[True]*16,elapsed_seconds=.1,input_tokens=20,output_tokens=10)
        for r in b['records']:
            r.update(lowest_confidence=float(i),bottom10_confidence=float(i),feature_input_tokens=30,
                     feature_seconds_share=.1,native_generation_seconds_share=.2,
                     extraction_inline={'input_tokens':20,'output_tokens':3,'seconds_share':.1},
                     extraction_final={'input_tokens':20,'output_tokens':3,'seconds_share':.1})
        result.append(b)
    return result


def test_matched_fit_lock_evaluation_and_transfer(monkeypatch,tmp_path):
    import pgm_asc.experiments.study_fit as fitting
    import pgm_asc.evaluation.study as evaluation
    from pgm_asc.models.study import ARMS
    monkeypatch.setattr(fitting,'load',fake_banks);monkeypatch.setattr(evaluation,'load',fake_banks)
    monkeypatch.setattr(fitting,'ARMS',{k:ARMS[k] for k in ['mixture','alpha1','random','candidate_confidence','deepsets_embedding']})
    cfg={'run_dir':str(tmp_path/'math'),'data_dir':str(tmp_path/'cohorts'),'model_tag':'toy','model_id':'fake','dataset':'math',
         'rounded_parameters':3e9,'train_seeds':[42,43,44],
         'train':{'lr':.001,'weight_decay':.001,'epochs':2,'patience':20,'question_batch':4}}
    cp=str(tmp_path/'config.json');write_json(cp,cfg)
    fitting.select(cfg,cp);lock=read_json(Path(cfg['run_dir'])/'selection_lock.json');verify_lock(lock)
    assert len(lock['methods'])==18 and lock['holdout_accessed'] is False
    result=evaluation.evaluate(cfg,'confirm')
    assert result['questions']==4 and len(result['primary_structure_contrasts_seed42'])==4
    assert all(0<=v['holm_p_four_primary_structure_contrasts']<=1 for v in result['primary_structure_contrasts_seed42'].values())
    assert result['summary']['candidate_confidence_s42__selected']['feature_input_tokens']>0
    transfer=cfg|{'run_dir':str(tmp_path/'omni'),'dataset':'omnimath','transfer_from':cfg['run_dir']}
    tp=str(tmp_path/'transfer.json');write_json(tp,transfer);fitting.select(transfer,tp)
    tl=read_json(Path(transfer['run_dir'])/'selection_lock.json');verify_lock(tl)
    assert tl['methods']==lock['methods'] and tl['holdout_accessed'] is False
    assert evaluation.holm({'a':.001,'b':.02,'c':.04,'d':.5})=={'a':.004,'b':.06,'c':.08,'d':.5}
