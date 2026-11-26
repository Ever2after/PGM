"""P1 nesting, integration, label isolation and shared latent continuations."""
import copy
import numpy as np
import pytest
import torch

from pgm_asc.hierarchical.model import KEYS, Statistics, forecast, infer, latent_grid, readable, tensors
from pgm_asc.hierarchical.train import fit_one, integration_check
from pgm_asc.objectives.train import Counts, SPEC
from pgm_asc.v2.model import Grid, from_json, log_theta, posterior


def bank():
    ids = np.array([[0,1,0,2]*4, [0,1]*8, [0]*16, list(range(16))])
    valid = np.ones_like(ids, bool)
    valid[0, [0,3,5]] = False
    correct = np.zeros((4,16), bool)
    correct[0,1] = True
    correct[1,0] = True
    return dict(ids=ids, valid=valid, correct=correct, question_id=np.array(['a','b','c','d']))


def test_nested_m1_likelihood_posterior_and_gradients_with_invalids():
    a = bank()
    raw = dict(zip(KEYS['M1'], [.2,-.4,.7,-.3]))
    p, old = tensors(raw, True), tensors(raw, True)
    st = Statistics(a)
    m = st.marginal(p, 'M1')
    native = Counts(a).marginal(old, Grid())
    torch.testing.assert_close(m, native, atol=1e-9, rtol=1e-10)
    j = st.joint_loss(p, 'M1', a)
    jo,_ = Counts(a).losses(old, Grid(), need_prefix=False)
    for g,h in zip(torch.autograd.grad(j, list(p.values())), torch.autograd.grad(jo,list(old.values()))):
        torch.testing.assert_close(g,h,atol=1e-9,rtol=1e-8)
    base = infer(p,'M1',a)
    ref = posterior(old,SPEC,a,{})
    np.testing.assert_allclose(base.post,ref.post,atol=1e-11)
    for variant in ('P1_linked','P1_independent','constant_theta'):
        r = {k: raw.get(k,0.) for k in KEYS[variant]}
        if variant in ('P1_independent','constant_theta'):
            reference = infer(tensors({**raw,'t1':0.}),'M1',a)
        else:
            reference = base
        nested = infer(tensors(r),variant,a,21)
        np.testing.assert_allclose(nested.post,reference.post,atol=1e-11)
        for i in range(4):
            np.testing.assert_allclose(nested.next_answer(i,8),reference.next_answer(i,8),atol=1e-11)


def test_normal_quadrature_moments_and_independent_prior():
    raw = dict(log_a=.1,log_b=-.3,t0=.2,sigma_raw=.7)
    g,lt,lp,_ = latent_grid(tensors(raw),'P1_independent',61)
    prob = lp.exp().numpy()
    assert prob.sum() == pytest.approx(1.)
    u = lt.numpy()-float(lt[0].mean())
    sd = readable(raw)['sigma_theta']
    assert np.sum(prob*u) == pytest.approx(0.,abs=1e-12)
    assert np.sum(prob*u*u) == pytest.approx(sd**2)
    cov = np.sum(prob*g.q.numpy()[:,None]*lt.numpy()) - np.sum(prob*g.q.numpy()[:,None])*np.sum(prob*lt.numpy())
    assert cov == pytest.approx(0.,abs=1e-12)
    _,linked,lpl,_ = latent_grid(tensors({**raw,'t1':1.}),'P1_linked',61)
    prob=lpl.exp().numpy()
    cov = np.sum(prob*g.q.numpy()[:,None]*linked.numpy()) - np.sum(prob*g.q.numpy()[:,None])*np.sum(prob*linked.numpy())
    assert cov > 0


def test_inference_and_forecast_do_not_read_labels_and_gain_is_paired():
    a = bank()
    raw = dict(log_a=.1,log_b=-.3,t0=.2,t1=.8,sigma_raw=.7)
    p = infer(tensors(raw),'P1_linked',{k:v for k,v in a.items() if k!='correct'},21)
    changed = {**a,'correct':~a['correct']}
    other = infer(tensors(raw),'P1_linked',changed,21)
    np.testing.assert_array_equal(p.post,other.post)
    np.testing.assert_allclose(p.post.sum(-1),1)
    f=forecast(a,p,[1,2,8,16],[1,2,4],64,7)
    f2=forecast(changed,other,[1,2,8,16],[1,2,4],64,7)
    for k in f:np.testing.assert_array_equal(f[k],f2[k])
    np.testing.assert_allclose(f['gain'],f['repair']-f['damage'],equal_nan=True)
    assert np.all(f['gain'][a['valid'][:,0],0,0]==0)
    for i in range(4):assert p.next_answer(i,8).sum()==pytest.approx(1.)


def test_supplied_latent_draws_reuse_identical_native_simulation():
    from pgm_asc.diagnostics.predictive import simulate
    joint=np.array([[.1,.2],[.3,.4]]+[[0.,0.]]*15)
    q,theta=np.array([.2,.7]),np.array([.3,2.])
    counts=np.r_[3.,2.,np.zeros(14)]
    first=np.r_[-2,-1,np.full(14,1000)]
    a,b=np.random.default_rng(9),np.random.default_rng(9)
    old=simulate(joint,counts,first,q,theta,[1,2,4],a,128,True)
    draws=b.choice(joint.size,128,p=joint.ravel())
    latent=(draws//2,q[draws%2],theta[draws%2])
    new=simulate(None,counts,first,None,None,[1,2,4],b,128,True,latent_draws=latent)
    assert old==new


def test_hierarchical_cdf_draws_equal_flat_joint_cdf():
    a=bank()
    p=infer(tensors(dict(log_a=.1,log_b=-.3,t0=.2,t1=.8,sigma_raw=.7)),'P1_linked',a,21)
    i,t=0,8
    sid=np.maximum(p.statistics.map[i,t-1],0)
    conditional=np.diff(np.c_[np.zeros(len(p.cdf)),p.cdf],axis=1)
    joint=p.post[i,t-1,:,None]*conditional[sid]
    u=np.random.default_rng(3).random(1000)
    flat=np.searchsorted(joint.ravel().cumsum(),u)
    c,q,theta=p.draw(i,t,u)
    np.testing.assert_array_equal(c,flat//joint.shape[1])
    np.testing.assert_array_equal(q,p.q[flat%joint.shape[1]])
    np.testing.assert_array_equal(theta,p.theta[flat%joint.shape[1]])


def test_quadrature_converges_and_random_effect_changes_likelihood():
    a=bank()
    raw=dict(log_a=.1,log_b=-.3,t0=.2,t1=.8,sigma_raw=.7)
    result=integration_check(raw,'P1_linked',a,a,31,61,[1,2,4,8,16],3.)
    assert result['max_error']<1e-6
    s=Statistics(a)
    assert abs(float(s.joint_loss(tensors(raw),'P1_linked',a,61))-
               float(s.joint_loss(tensors({**raw,'sigma_raw':0.}),'P1_linked',a,61)))>1e-5


@pytest.mark.parametrize('variant',list(KEYS))
def test_each_arm_fits_joint_and_parameter_count(variant):
    from pgm_asc.hierarchical import CONFIG
    settings={**CONFIG['training'],'max_iter':20}
    fit=fit_one(bank(),variant,settings,0,15)
    assert fit['parameters']==len(KEYS[variant])
    before=float(Statistics(bank()).joint_loss(tensors(fit['initial']),variant,bank(),15))
    assert fit['joint_nll_per_response']<before


def test_tiny_end_to_end_protocol_fit_lock_report_and_resume(tmp_path, monkeypatch):
    import json
    from pgm_asc import config
    from pgm_asc.hierarchical import CONFIG
    from pgm_asc.hierarchical import run as runner
    artifact=tmp_path/'artifacts'
    (artifact/'v2').mkdir(parents=True)
    arrays=bank()
    np.savez_compressed(artifact/'v2'/'toy_math.npz',
        **{f'{s}.{k}':v for s in ('train','dev','test') for k,v in arrays.items()})
    monkeypatch.setattr(config,'ARTIFACTS',artifact)
    monkeypatch.setattr(config,'RESULTS',tmp_path/'results')
    study=copy.deepcopy(CONFIG)
    study['models'],study['datasets']=['toy'],['math']
    study['training'].update(max_iter=12,restart_seeds=[0],sigma_starts=[.25])
    study['integration'].update(normal_nodes=[15,31,61],tolerance=.1)
    study['evaluation'].update(simulations_per_seed=16,mc_seeds=[7,8],bootstrap_replicates=10)
    out=tmp_path/'report'
    paths=runner.run(workers=1,out=out,study=study)
    assert len(paths)==1
    lock=json.loads((out/'selection'/'toy_math.json').read_text())
    result=json.loads((out/'cells'/'toy_math.json').read_text())
    assert lock['selection_locked_at']<=result['evaluation_started_at']
    assert set(result['splits']['test']['models'])==set(KEYS)
    assert [result['parameters'][n]['count'] for n in KEYS]==[4,3,4,5,5]
    assert (out/'report.html').exists() and (out/'figures'/'gain.png').exists()
    assert (out/'quadrature_checks.csv').exists()
    def fail(*args,**kwargs):raise AssertionError('resume must not refit or simulate')
    monkeypatch.setattr(runner,'fit_one',fail)
    monkeypatch.setattr(runner,'forecast',fail)
    again=runner.run(workers=1,out=out,study=study)
    assert paths==again
    with pytest.raises(ValueError,match='Protocol/code differs'):
        changed=copy.deepcopy(study)
        changed['training']['ridge']=.01
        runner.run(workers=1,out=out,study=changed)


def test_no_link_factorizes_q_theta_conditional_on_answer_hypothesis():
    a=bank()
    p=infer(tensors(dict(log_a=.1,log_b=-.3,t0=.2,sigma_raw=1.2)),'P1_independent',a,31)
    probability=np.diff(np.c_[np.zeros(len(p.cdf)),p.cdf],axis=1).reshape(-1,80,31)
    product=probability.sum(2)[:,:,None]*probability.sum(1)[:,None,:]
    np.testing.assert_allclose(probability,product,atol=1e-13)
