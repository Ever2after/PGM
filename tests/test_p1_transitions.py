"""Exact event learning preserves P1 and uses suffix labels only as targets."""
import copy
import numpy as np
import pytest
import torch

from pgm_asc.hierarchical.model import KEYS,infer,tensors
from pgm_asc.objectives.train import Objective
from pgm_asc.p1_objectives.train import Counts as OldCounts
from pgm_asc.p1_transitions.train import Counts,fit_one
from pgm_asc.p1_transitions.kernel import conditional_success


def bank():
    ids=np.array([[15,0,0,1]*4,[0,1]*8,[0]*16,list(range(16))])
    valid=ids!=15
    correct=np.zeros((4,16),bool);correct[0,1]=True;correct[1,0]=True
    return dict(ids=ids,valid=valid,correct=correct,question_id=np.array(['a','b','c','d']))


@pytest.mark.parametrize('variant',['P1_independent','P1_linked'])
def test_event_integrals_equal_native_posterior_times_conditional_kernel(variant):
    a=bank();counts=Counts(a)
    raw={k:v for k,v in dict(log_a=.2,log_b=-.4,t0=.7,t1=-.3,sigma_raw=.8).items() if k in KEYS[variant]}
    p=tensors(raw,True);native=infer(p,variant,a,31)
    events,current=counts.event_probabilities(p,variant,31)
    events,current=events.detach().numpy(),current.detach().numpy()
    conditional=np.diff(np.c_[np.zeros(len(native.cdf)),native.cdf],axis=1)
    for qi in range(4):
        for ei,(t,h) in enumerate(counts.pairs_th):
            observed=native.statistics.count[qi,t-1,:16]
            first=np.full(16,1000)
            for j in range(t):
                if a['valid'][qi,j]:first[a['ids'][qi,j]]=min(first[a['ids'][qi,j]],j-t)
            leader=int(np.where(observed==observed.max(),first,1000).argmin()) if observed.sum() else -1
            r=d=now=0.
            for hyp in np.flatnonzero(native.post[qi,t-1]):
                sid=native.statistics.map[qi,t-1,hyp]
                win=conditional_success(observed,first,int(hyp),h,native.q,native.theta)
                success=float((conditional[sid]*win).sum())
                weight=native.post[qi,t-1,hyp]
                if hyp==leader:d+=weight*(1-success);now+=weight
                else:r+=weight*success
            np.testing.assert_allclose(events[qi,ei],[r,d,1-r-d],atol=3e-11)
            assert current[qi,ei]==pytest.approx(now,abs=2e-12)
    np.testing.assert_allclose(events.sum(2),1.,atol=1e-11)
    assert np.all(events>=0)
    # With one valid observed answer, one extra answer cannot overturn it.
    index=counts.pairs_th.index((1,1))
    np.testing.assert_array_equal(events[1:,index,:2],0.)


def test_event_probabilities_ignore_gold_and_later_suffix_and_gradients_match_finite_difference():
    a=bank();raw=dict(log_a=.2,log_b=-.4,t0=.7,t1=-.3,sigma_raw=.8)
    c=Counts(a);p=tensors(raw,True)
    event,_=c.event_probabilities(p,'P1_linked',31)
    changed={**a,'correct':~a['correct']}
    other,_=Counts(changed).event_probabilities(p,'P1_linked',31)
    np.testing.assert_array_equal(event.detach().numpy(),other.detach().numpy())
    changed={k:v.copy() for k,v in a.items()};changed['ids'][:,4:]=0;changed['valid'][:,4:]=True
    other,_=Counts(changed).event_probabilities(p,'P1_linked',31)
    keep=[i for i,(t,h) in enumerate(c.pairs_th) if t<=4]
    np.testing.assert_allclose(event.detach().numpy()[:,keep],other.detach().numpy()[:,keep],atol=1e-12)
    loss=c.components(p,'P1_linked',31)[2]
    gradients=torch.autograd.grad(loss,list(p.values()))
    eps=1e-5
    for k,g in zip(p,gradients):
        plus=c.components(tensors({**raw,k:raw[k]+eps}),'P1_linked',31)[2]
        minus=c.components(tensors({**raw,k:raw[k]-eps}),'P1_linked',31)[2]
        assert float(g)==pytest.approx(float((plus-minus)/(2*eps)),abs=2e-8,rel=2e-6)


@pytest.mark.parametrize('beta',[0.,.1,1.,10.])
def test_proper_transition_nll_mixed_scaling_and_parameter_count(beta):
    from pgm_asc.p1_transitions import CONFIG
    from pgm_asc.p1_objectives.train import fit_one as old_fit
    settings={**CONFIG['training'],'max_iter':20}
    a=bank();counts=Counts(a)
    f=fit_one(counts,'P1_linked',settings,0,1.,beta,'uniform',15)
    assert f['parameters']==5
    assert f['data_loss']==pytest.approx(((f['joint_nll_per_response']+f['prefix_nll'])/2+beta*f['transition_nll'])/(1+beta))
    if beta==0:
        old=old_fit(OldCounts(a),'P1_linked',Objective('mixed','uniform',1.),settings,0,15)
        assert f['penalized_loss']==pytest.approx(old['penalized_loss'],abs=1e-10)


def test_exact_sensitivity_scores_equal_training_transition_nll_and_preserve_gain_identity():
    from pgm_asc.hierarchical.model import forecast
    from pgm_asc.p1_transitions.evaluation import exact_forecast,transition_nll
    a=bank();counts=Counts(a)
    raw=dict(log_a=.2,log_b=-.4,t0=.7,t1=-.3,sigma_raw=.8);p=tensors(raw)
    native=infer(p,'P1_linked',a,31)
    prefixes,horizons=[1,2,4,8,16],[1,2,4,8]
    mc=forecast(a,native,prefixes,horizons,64,7)
    exact=exact_forecast(counts,p,'P1_linked',31,3.,prefixes,horizons,mc)
    np.testing.assert_allclose(exact['gain'],exact['repair']-exact['damage'],equal_nan=True)
    np.testing.assert_allclose(exact['future_reference_correct'],exact['current_correct_mc']+exact['gain'],atol=1e-12,equal_nan=True)
    expected=counts.components(p,'P1_linked',31)[2]
    assert transition_nll(exact,a,prefixes,horizons).mean()==pytest.approx(float(expected),abs=1e-11)


def test_full_transition_pipeline_locks_frozen_comparators_and_resumes(tmp_path,monkeypatch):
    import json
    from pgm_asc import config
    from pgm_asc.hierarchical import CONFIG as HCONFIG
    from pgm_asc.hierarchical.run import run as hr
    from pgm_asc.p1_objectives import CONFIG as OCONFIG
    from pgm_asc.p1_objectives.run import run as orun
    from pgm_asc.p1_predictive import CONFIG as PCONFIG
    from pgm_asc.p1_predictive.run import run as prun
    from pgm_asc.p1_transitions import CONFIG
    from pgm_asc.p1_transitions import run as runner
    artifact=tmp_path/'artifacts';(artifact/'v2').mkdir(parents=True)
    a=bank();np.savez_compressed(artifact/'v2'/'toy_math.npz',**{f'{s}.{k}':v for s in ('train','dev','test') for k,v in a.items()})
    monkeypatch.setattr(config,'ARTIFACTS',artifact);monkeypatch.setattr(config,'RESULTS',tmp_path/'results')
    h=copy.deepcopy(HCONFIG);h['models'],h['datasets']=['toy'],['math']
    h['training'].update(max_iter=10,restart_seeds=[0],sigma_starts=[.25])
    h['integration'].update(normal_nodes=[61,121,181],tolerance=.01)
    h['evaluation'].update(simulations_per_seed=16,mc_seeds=[7],bootstrap_replicates=10)
    hr(workers=1,study=h)
    o=copy.deepcopy(OCONFIG);o['models'],o['datasets']=['toy'],['math']
    o['training'].update(max_iter=10,restart_seeds=[0],sigma_starts=[.25],weightings=['uniform'],lambda_grid=[.1,1.])
    o['integration'],o['evaluation']=h['integration'],h['evaluation'];orun(workers=1,study=o)
    p=copy.deepcopy(PCONFIG);p['models'],p['datasets']=['toy'],['math'];p['training'].update(o['training'])
    p['training']['predictive_weights']=[1.];p['integration'],p['evaluation']=h['integration'],h['evaluation']
    prun(workers=1,study=p)
    study=copy.deepcopy(CONFIG);study['models'],study['datasets']=['toy'],['math'];study['training'].update(p['training'])
    study['training']['transition_weights']=[.1];study['integration'],study['evaluation']=h['integration'],h['evaluation']
    paths=runner.run(workers=1,study=study)
    out=config.RESULTS/'p1_transitions';r=json.loads((out/'cells'/'toy_math.json').read_text())
    lock=json.loads((out/'selection'/'toy_math.json').read_text());validation=json.loads((out/'validation.json').read_text())
    assert len(r['splits']['test']['models'])==6
    assert lock['selection_locked_at']<=r['evaluation_started_at']
    assert validation['frozen_arrays_identical'] and validation['frozen_summary_max_difference']==0
    assert r['splits']['test']['exact_paired_contrasts']
    assert (out/'exact_transition_macro.csv').exists() and (out/'report.html').exists()
    for v in study['variants']:
        base=lock['fits'][f'{v}__mixed_uniform']['objective']['mixing']
        assert lock['fits'][f'{v}__transition_uniform_beta_0p1']['objective']['mixing']==base
    def fail(*args,**kwargs):raise AssertionError('resume should not refit or simulate')
    monkeypatch.setattr(runner,'fit_one',fail);monkeypatch.setattr(runner,'forecast',fail)
    assert runner.run(workers=1,study=study)==paths
