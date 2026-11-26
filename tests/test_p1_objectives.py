"""Objective changes must preserve the fixed P1 joint distribution and inputs."""
import copy
import json

import numpy as np
import pytest
import torch

from pgm_asc.hierarchical.model import KEYS,infer,tensors
from pgm_asc.objectives.train import Counts as M1Counts, Objective, prefix_weights
from pgm_asc.p1_objectives.train import Counts,dev_score,fit_one,integration_check
from pgm_asc.v2.model import Grid


def bank():
    ids=np.array([[0,1,0,2]*4,[0,1]*8,[0]*16,list(range(16))])
    valid=np.ones_like(ids,bool);valid[0,[0,3,5]]=False
    correct=np.zeros((4,16),bool);correct[0,1]=True;correct[1,0]=True
    return dict(ids=ids,valid=valid,correct=correct,question_id=np.array(['a','b','c','d']))


@pytest.mark.parametrize('variant',['P1_linked','P1_independent'])
@pytest.mark.parametrize('weighting',['uniform','v1'])
def test_prefix_losses_equal_native_posterior_and_nested_m1_gradients(variant,weighting):
    a=bank();counts=Counts(a)
    raw={k:v for k,v in dict(log_a=.2,log_b=-.4,t0=.7,t1=-.3,sigma_raw=.8).items() if k in KEYS[variant]}
    p=tensors(raw,True)
    j,c=counts.losses(p,variant,31,weighting)
    posterior=infer(p,variant,a,31).post
    target=counts.stats.hyp[counts.target.numpy()].reshape(-1,16)
    native=-np.log(np.take_along_axis(posterior,target[...,None],2)[...,0])
    assert float(c.detach())==pytest.approx(float((native*prefix_weights(weighting)).sum(1).mean()),abs=1e-11)
    raw['sigma_raw']=0.
    p=tensors(raw,True)
    ref={k:raw.get(k,0.) for k in ('log_a','log_b','t0','t1')}
    r=tensors(ref,True)
    j,c=counts.losses(p,variant,31,weighting)
    jr,cr=M1Counts(a).losses(r,Grid(),weighting)
    torch.testing.assert_close(j,jr,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(c,cr,atol=1e-10,rtol=1e-10)
    actual=torch.autograd.grad(c,list(p.values()))
    expected=dict(zip(r,torch.autograd.grad(cr,list(r.values()))))
    for k,g in zip(p,actual):
        if k!='sigma_raw':torch.testing.assert_close(g,expected[k],atol=1e-9,rtol=1e-8)


@pytest.mark.parametrize('variant',['P1_linked','P1_independent'])
@pytest.mark.parametrize('kind,mixing',[('joint',None),('prefix',None),('mixed',1.)])
def test_fit_modes_keep_graph_and_common_initialization(variant,kind,mixing):
    from pgm_asc.p1_objectives import CONFIG
    from pgm_asc.hierarchical.train import fit_one as old_fit
    settings={**CONFIG['training'],'max_iter':20}
    obj=Objective(kind,'uniform',mixing)
    f=fit_one(Counts(bank()),variant,obj,settings,0,15)
    assert set(f['params'])==set(KEYS[variant])
    assert f['parameters']==len(KEYS[variant])
    old=old_fit(bank(),variant,settings,0,15)
    assert f['initial']==old['initial']
    if kind=='joint':
        assert f['params']==pytest.approx(old['params'],abs=1e-8)
        assert f['penalized_loss']==pytest.approx(old['penalized_loss'],abs=1e-11)
    if kind=='mixed':
        assert f['data_loss']==pytest.approx((f['joint_nll_per_response']+f['prefix_nll'])/2)


def test_quadrature_checks_training_prefix_and_label_free_inference():
    a=bank();raw=dict(log_a=.2,log_b=-.4,t0=.7,t1=-.3,sigma_raw=.8)
    counts=Counts(a)
    check=integration_check(raw,'P1_linked',Objective('mixed','v1',.1),counts,counts,a,31,61,[1,2,4,8,16],3.)
    assert check['max_error']<1e-6
    assert 'train_prefix' in check['errors']
    p=infer(tensors(raw),'P1_linked',a,31)
    a['correct']=~a['correct']
    other=infer(tensors(raw),'P1_linked',a,31)
    np.testing.assert_array_equal(p.post,other.post)


def test_tiny_pipeline_frozen_joint_selection_lock_and_resume(tmp_path,monkeypatch):
    from pgm_asc import config
    from pgm_asc.hierarchical import CONFIG as BASE_CONFIG
    from pgm_asc.hierarchical.run import run as baseline_run
    from pgm_asc.p1_objectives import CONFIG
    from pgm_asc.p1_objectives import run as runner
    artifact=tmp_path/'artifacts';(artifact/'v2').mkdir(parents=True)
    a=bank()
    np.savez_compressed(artifact/'v2'/'toy_math.npz',
        **{f'{s}.{k}':v for s in ('train','dev','test') for k,v in a.items()})
    monkeypatch.setattr(config,'ARTIFACTS',artifact)
    monkeypatch.setattr(config,'RESULTS',tmp_path/'results')
    base=copy.deepcopy(BASE_CONFIG)
    base['models'],base['datasets']=['toy'],['math']
    base['training'].update(max_iter=12,restart_seeds=[0],sigma_starts=[.25])
    base['integration'].update(normal_nodes=[61,121,181],tolerance=.01)
    base['evaluation'].update(simulations_per_seed=16,mc_seeds=[7,8],bootstrap_replicates=10)
    baseline_run(workers=1,out=config.RESULTS/'hierarchical_crp',study=base)
    study=copy.deepcopy(CONFIG)
    study['models'],study['datasets']=['toy'],['math']
    study['training'].update(max_iter=12,restart_seeds=[0],sigma_starts=[.25],weightings=['uniform'],lambda_grid=[.1,1.])
    study['integration']=base['integration'];study['evaluation']=base['evaluation']
    out=config.RESULTS/'p1_objectives'
    paths=runner.run(workers=1,out=out,study=study)
    result=json.loads((out/'cells'/'toy_math.json').read_text())
    selected=json.loads((out/'selection'/'toy_math.json').read_text())
    assert selected['selection_locked_at']<=result['evaluation_started_at']
    assert len(result['splits']['test']['models'])==6
    for v in study['variants']:
        chosen=selected['selected'][f'{v}__mixed_uniform']
        candidates=[n for n in selected['fits'] if n.startswith(f'{v}__mixed_uniform_lambda')]
        assert chosen==min(candidates,key=lambda n:selected['fits'][n]['best']['dev_unique_nll'])
    validation=json.loads((out/'validation.json').read_text())
    assert validation['joint_arrays_identical'] and validation['joint_summary_max_difference']==0
    assert validation['max_gain_identity_error']<1e-12
    assert (out/'report.html').exists() and (out/'figures'/'gain_uniform.png').exists()
    def fail(*args,**kwargs):raise AssertionError('cached resume must not fit or simulate')
    monkeypatch.setattr(runner,'fit_one',fail);monkeypatch.setattr(runner,'forecast',fail)
    assert runner.run(workers=1,out=out,study=study)==paths
    changed=copy.deepcopy(study);changed['training']['ridge']=.01
    with pytest.raises(ValueError,match='Protocol/code differs'):
        runner.run(workers=1,out=out,study=changed)
