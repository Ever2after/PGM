"""Predictive training must equal the native label-free P1 answer generator."""
import copy
import numpy as np
import pytest
import torch

from pgm_asc.hierarchical.model import KEYS,infer,tensors
from pgm_asc.p1_predictive.train import Counts,data_loss,fit_one
from pgm_asc.p1_objectives.train import Counts as OldCounts
from pgm_asc.objectives.train import Objective


def bank():
    ids=np.array([[0,1,0,2]*4,[0,1]*8,[0]*16,list(range(16))])
    valid=np.ones_like(ids,bool);valid[0,[0,3,5]]=False
    correct=np.zeros((4,16),bool);correct[0,1]=True;correct[1,0]=True
    return dict(ids=ids,valid=valid,correct=correct,question_id=np.array(['a','b','c','d']))


@pytest.mark.parametrize('variant',['P1_independent','P1_linked'])
def test_predictive_target_probabilities_equal_native_generator_at_all_prefixes(variant):
    a=bank()
    raw={k:v for k,v in dict(log_a=.2,log_b=-.4,t0=.7,t1=-.3,sigma_raw=.8).items() if k in KEYS[variant]}
    p=tensors(raw,True);counts=Counts(a)
    j,c,n,_=counts.components(p,variant,31)
    native=infer(p,variant,a,31)
    expected=np.zeros((4,15))
    for i in range(4):
        for t in range(1,16):
            existing=np.any((a['ids'][i,:t]==a['ids'][i,t])&a['valid'][i,:t])
            target=int(a['ids'][i,t]) if existing else 16
            if not a['valid'][i,t]:target=17
            expected[i,t-1]=native.next_answer(i,t)[target]
    _,_,logp=counts.state_values(p,variant,31)
    np.testing.assert_allclose(logp.detach().numpy(),np.log(np.clip(expected,1e-12,1)),atol=2e-11)
    assert float(n.detach())==pytest.approx(-np.log(np.clip(expected,1e-12,1)).mean(),abs=1e-11)
    for w in ('uniform','v1'):
        jo,co=OldCounts(a).losses(p,variant,31,w)
        j,c,_,_=counts.components(p,variant,31,w)
        torch.testing.assert_close(j,jo,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(c,co,atol=1e-10,rtol=1e-10)
        actual=torch.autograd.grad(j+c,list(p.values()),retain_graph=True)
        original=torch.autograd.grad(jo+co,list(p.values()),retain_graph=True)
        for x,y in zip(actual,original):torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-8)


def test_predictive_loss_ignores_gold_and_later_suffix_and_has_correct_gradient():
    a=bank();raw=dict(log_a=.2,log_b=-.4,t0=.7,t1=-.3,sigma_raw=.8)
    p=tensors(raw,True)
    _,_,logp=Counts(a).state_values(p,'P1_linked',31)
    changed={k:v.copy() for k,v in a.items() if k!='correct'}
    changed['ids'][:,8:]=0
    _,_,other=Counts(changed).state_values(p,'P1_linked',31)
    np.testing.assert_allclose(logp.detach().numpy()[:,:7],other.detach().numpy()[:,:7],atol=1e-12)
    changed={**a,'correct':~a['correct']}
    _,_,other=Counts(changed).state_values(p,'P1_linked',31)
    np.testing.assert_array_equal(logp.detach().numpy(),other.detach().numpy())
    loss=-logp.mean();gradient=torch.autograd.grad(loss,list(p.values()))
    eps=1e-5
    for k,g in zip(p,gradient):
        plus,minus={**raw,k:raw[k]+eps},{**raw,k:raw[k]-eps}
        lp=Counts(a).state_values(tensors(plus),'P1_linked',31)[-1]
        lm=Counts(a).state_values(tensors(minus),'P1_linked',31)[-1]
        numerical=float((-lp.mean()+lm.mean())/(2*eps))
        assert float(g)==pytest.approx(numerical,abs=2e-8,rel=1e-6)


@pytest.mark.parametrize('gamma',[0.,.1,1.,10.])
def test_normalized_loss_and_fitting_keep_parameter_count(gamma):
    from pgm_asc.p1_predictive import CONFIG
    settings={**CONFIG['training'],'max_iter':20}
    counts=Counts(bank())
    f=fit_one(counts,'P1_linked',settings,0,1.,gamma,'uniform',15)
    assert f['parameters']==5
    assert f['data_loss']==pytest.approx(((f['joint_nll_per_response']+f['prefix_nll'])/2+gamma*f['predictive_nll'])/(1+gamma))
    if gamma==0:
        from pgm_asc.p1_objectives.train import fit_one as old_fit
        old=old_fit(OldCounts(bank()),'P1_linked',Objective('mixed','uniform',1.),settings,0,15)
        assert f['initial']==old['initial']
        assert f['penalized_loss']==pytest.approx(old['penalized_loss'],abs=1e-10)


def test_three_stage_pipeline_preserves_all_frozen_references_and_resumes(tmp_path,monkeypatch):
    import json
    from pgm_asc import config
    from pgm_asc.hierarchical import CONFIG as HCONFIG
    from pgm_asc.hierarchical.run import run as hierarchical_run
    from pgm_asc.p1_objectives import CONFIG as OCONFIG
    from pgm_asc.p1_objectives.run import run as objective_run
    from pgm_asc.p1_predictive import CONFIG
    from pgm_asc.p1_predictive import run as runner
    artifact=tmp_path/'artifacts';(artifact/'v2').mkdir(parents=True)
    a=bank()
    np.savez_compressed(artifact/'v2'/'toy_math.npz',**{f'{s}.{k}':v for s in ('train','dev','test') for k,v in a.items()})
    monkeypatch.setattr(config,'ARTIFACTS',artifact);monkeypatch.setattr(config,'RESULTS',tmp_path/'results')
    h=copy.deepcopy(HCONFIG);h['models'],h['datasets']=['toy'],['math']
    h['training'].update(max_iter=10,restart_seeds=[0],sigma_starts=[.25])
    h['integration'].update(normal_nodes=[61,121,181],tolerance=.01)
    h['evaluation'].update(simulations_per_seed=16,mc_seeds=[7,8],bootstrap_replicates=10)
    hierarchical_run(workers=1,study=h)
    o=copy.deepcopy(OCONFIG);o['models'],o['datasets']=['toy'],['math']
    o['training'].update(max_iter=10,restart_seeds=[0],sigma_starts=[.25],weightings=['uniform'],lambda_grid=[.1,1.])
    o['integration'],o['evaluation']=h['integration'],h['evaluation']
    objective_run(workers=1,study=o)
    study=copy.deepcopy(CONFIG);study['models'],study['datasets']=['toy'],['math']
    study['training'].update(o['training']);study['training']['predictive_weights']=[.1]
    study['integration'],study['evaluation']=h['integration'],h['evaluation']
    paths=runner.run(workers=1,study=study)
    out=config.RESULTS/'p1_predictive'
    result=json.loads((out/'cells'/'toy_math.json').read_text())
    lock=json.loads((out/'selection'/'toy_math.json').read_text())
    assert lock['selection_locked_at']<=result['evaluation_started_at']
    assert len(result['splits']['test']['models'])==8
    for v in study['variants']:
        mixed=lock['fits'][f'{v}__mixed_uniform']
        new=lock['fits'][f'{v}__predictive_uniform_gamma_0p1']
        assert mixed['objective']['mixing']==new['objective']['mixing']
    validation=json.loads((out/'validation.json').read_text())
    assert validation['frozen_arrays_identical'] and validation['frozen_summary_max_difference']==0
    assert (out/'posterior_retention.csv').exists() and (out/'report.html').exists()
    def fail(*args,**kwargs):raise AssertionError('resume must not fit or simulate')
    monkeypatch.setattr(runner,'fit_one',fail);monkeypatch.setattr(runner,'forecast',fail)
    assert runner.run(workers=1,study=study)==paths
    changed=copy.deepcopy(study);changed['training']['predictive_weights']=[1.]
    with pytest.raises(ValueError,match='Protocol/code differs'):
        runner.run(workers=1,study=changed)


def test_valid_transitions_also_equal_marginal_evidence_ratios():
    a=bank();p=tensors(dict(log_a=.2,log_b=-.4,t0=.7,t1=-.3,sigma_raw=.8))
    _,normalizer,logp=Counts(a).state_values(p,'P1_linked',31)
    ratio=normalizer.reshape(-1,16)[:,1:]-normalizer.reshape(-1,16)[:,:-1]
    valid=torch.tensor(a['valid'][:,1:])
    torch.testing.assert_close(ratio[valid],logp[valid],atol=1e-11,rtol=1e-11)
    # The legacy likelihood assigns (1-q) to invalid observations, whereas its
    # future generator has no invalid branch. Predictive NLL uses the generator.
    assert torch.all(ratio[~valid]>logp[~valid])
