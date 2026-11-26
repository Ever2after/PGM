"""Brier fitting must change the loss without changing P1 event probabilities."""
import copy
import numpy as np
import pytest
import torch

from pgm_asc.hierarchical.model import KEYS,infer,forecast,tensors
from pgm_asc.objectives.train import Objective
from pgm_asc.p1_objectives.train import Counts as MixedCounts
from pgm_asc.p1_transitions.train import Counts as NLLCounts
from pgm_asc.p1_transitions.evaluation import exact_forecast
from pgm_asc.p1_brier.train import Counts,fit_one
from pgm_asc.p1_brier.evaluation import diagnostics


def bank():
    ids=np.array([[15,0,0,1]*4,[0,1]*8,[0]*16,list(range(16))]);valid=ids!=15
    correct=np.zeros((4,16),bool);correct[0,1]=True;correct[1,0]=True
    return dict(ids=ids,valid=valid,correct=correct,question_id=np.array(['a','b','c','d']))


@pytest.mark.parametrize('variant',['P1_independent','P1_linked'])
def test_brier_equals_two_binary_scores_and_preserves_event_probabilities(variant):
    a=bank();c=Counts(a);old=NLLCounts(a)
    raw={k:v for k,v in dict(log_a=.2,log_b=-.4,t0=.7,t1=-.3,sigma_raw=.8).items() if k in KEYS[variant]}
    p=tensors(raw,True);event,_=c.event_probabilities(p,variant,31);reference,_=old.event_probabilities(p,variant,31)
    np.testing.assert_array_equal(event.detach().numpy(),reference.detach().numpy())
    truth=c.truth.reshape(4,-1).numpy();ep=event.detach().numpy()
    expected=(((ep[:,:,0]-(truth==0))**2).mean()+((ep[:,:,1]-(truth==1))**2).mean())/2
    j,cp,b,nll=c.components(p,variant,31)
    assert float(b.detach())==pytest.approx(expected,abs=1e-12)
    jo,co,_,no=old.components(p,variant,31)
    torch.testing.assert_close(j,jo);torch.testing.assert_close(cp,co);torch.testing.assert_close(nll,no)
    changed={k:v for k,v in a.items() if k!='correct'}
    other,_=Counts(changed).event_probabilities(p,variant,31)
    np.testing.assert_array_equal(event.detach().numpy(),other.detach().numpy())


def test_brier_gradient_finite_difference_and_gain_error_decomposition():
    a=bank();c=Counts(a);raw=dict(log_a=.2,log_b=-.4,t0=.7,t1=-.3,sigma_raw=.8)
    p=tensors(raw,True);b=c.components(p,'P1_linked',31)[2]
    grad=torch.autograd.grad(b,list(p.values()));eps=1e-5
    for k,g in zip(p,grad):
        plus=c.components(tensors({**raw,k:raw[k]+eps}),'P1_linked',31)[2]
        minus=c.components(tensors({**raw,k:raw[k]-eps}),'P1_linked',31)[2]
        assert float(g)==pytest.approx(float((plus-minus)/(2*eps)),abs=2e-9,rel=2e-6)
    native=infer(p,'P1_linked',a,31);prefixes=[1,2,4,8,16];horizons=[1,2,4,8]
    mc=forecast(a,native,prefixes,horizons,64,7)
    exact=exact_forecast(c,p,'P1_linked',31,3.,prefixes,horizons,mc)
    values=diagnostics(exact,a,prefixes,horizons)
    assert values['rd_mean_brier'].mean()==pytest.approx(float(b.detach()),abs=1e-11)
    assert np.abs(values['gain_decomposition_residual']).max()<1e-15


@pytest.mark.parametrize('beta',[0.,.1,1.,10.,100.])
def test_common_optimizer_scaling_and_parameter_count(beta):
    from pgm_asc.p1_brier import CONFIG
    from pgm_asc.p1_objectives.train import fit_one as mixed_fit
    settings={**CONFIG['training'],'max_iter':20};a=bank()
    f=fit_one(Counts(a),'P1_linked',settings,0,1.,beta,'uniform',15)
    assert f['parameters']==5 and 'transition_nll' not in f
    assert f['data_loss']==pytest.approx(((f['joint_nll_per_response']+f['prefix_nll'])/2+beta*f['transition_brier'])/(1+beta))
    if beta==0:
        old=mixed_fit(MixedCounts(a),'P1_linked',Objective('mixed','uniform',1.),settings,0,15)
        assert f['initial']==old['initial']
        assert f['penalized_loss']==pytest.approx(old['penalized_loss'],abs=1e-10)


def test_brier_pipeline_freezes_mixed_and_nll_references_and_resumes(tmp_path,monkeypatch):
    import json
    from pgm_asc import config
    from pgm_asc.hierarchical import CONFIG as HC
    from pgm_asc.hierarchical.run import run as hr
    from pgm_asc.p1_objectives import CONFIG as OC
    from pgm_asc.p1_objectives.run import run as orun
    from pgm_asc.p1_predictive import CONFIG as PC
    from pgm_asc.p1_predictive.run import run as prun
    from pgm_asc.p1_transitions import CONFIG as TC
    from pgm_asc.p1_transitions.run import run as trun
    from pgm_asc.p1_brier import CONFIG
    from pgm_asc.p1_brier import run as runner
    artifact=tmp_path/'artifacts';(artifact/'v2').mkdir(parents=True)
    a=bank();np.savez_compressed(artifact/'v2'/'toy_math.npz',**{f'{s}.{k}':v for s in ('train','dev','test') for k,v in a.items()})
    monkeypatch.setattr(config,'ARTIFACTS',artifact);monkeypatch.setattr(config,'RESULTS',tmp_path/'results')
    h=copy.deepcopy(HC);h['models'],h['datasets']=['toy'],['math']
    h['training'].update(max_iter=8,restart_seeds=[0],sigma_starts=[.25])
    h['integration'].update(normal_nodes=[61,121,181],tolerance=.01)
    h['evaluation'].update(simulations_per_seed=16,mc_seeds=[7],bootstrap_replicates=10);hr(workers=1,study=h)
    o=copy.deepcopy(OC);o['models'],o['datasets']=['toy'],['math']
    o['training'].update(max_iter=8,restart_seeds=[0],sigma_starts=[.25],weightings=['uniform'],lambda_grid=[.1,1.])
    o['integration'],o['evaluation']=h['integration'],h['evaluation'];orun(workers=1,study=o)
    p=copy.deepcopy(PC);p['models'],p['datasets']=['toy'],['math'];p['training'].update(o['training'])
    p['training']['predictive_weights']=[1.];p['integration'],p['evaluation']=h['integration'],h['evaluation'];prun(workers=1,study=p)
    t=copy.deepcopy(TC);t['models'],t['datasets']=['toy'],['math'];t['training'].update(p['training'])
    t['training']['transition_weights']=[1.];t['integration'],t['evaluation']=h['integration'],h['evaluation'];trun(workers=1,study=t)
    study=copy.deepcopy(CONFIG);study['models'],study['datasets']=['toy'],['math'];study['training'].update(t['training'])
    study['training']['brier_weights']=[1.];study['integration'],study['evaluation']=h['integration'],h['evaluation']
    paths=runner.run(workers=1,study=study)
    out=config.RESULTS/'p1_brier';r=json.loads((out/'cells'/'toy_math.json').read_text())
    lock=json.loads((out/'selection'/'toy_math.json').read_text());qa=json.loads((out/'validation.json').read_text())
    assert len(r['splits']['test']['models'])==6
    assert lock['selection_locked_at']<=r['evaluation_started_at']
    assert qa['frozen_arrays_identical'] and qa['frozen_summary_max_difference']==0
    assert (out/'gain_decomposition.csv').exists() and (out/'report.html').exists()
    for entry in r['splits']['test']['models'].values():
        for field in ('summary','exact_transition_summary'):
            assert abs(entry[field]['gain_decomposition_residual']['mean'])<1e-15
            expected=(entry[field]['repair_brier']['mean']+entry[field]['damage_brier']['mean'])/2
            assert entry[field]['rd_mean_brier']['mean']==pytest.approx(expected,abs=1e-14)
    def fail(*args,**kwargs):raise AssertionError('resume should not fit or simulate')
    monkeypatch.setattr(runner,'fit_one',fail);monkeypatch.setattr(runner,'forecast',fail)
    assert runner.run(workers=1,study=study)==paths
