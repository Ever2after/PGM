import copy
import json

import numpy as np
import pytest
import torch

from pgm_asc.hierarchical.model import infer,tensors
from pgm_asc.p1_stopping.model import CountPosterior,decisions,trace
from pgm_asc.v2 import CONFIG as V2


def bank():
    ids=np.array([[0,0,1,2]*4,[0,1]*8,[0]*16,[0,1,2,3]*4])
    valid=np.ones_like(ids,bool);valid[0,0]=False
    correct=np.zeros_like(valid);correct[:,0]=True
    return dict(ids=ids,valid=valid,correct=correct,question_id=np.array(['a','b','c','d']),tokens=np.ones_like(ids))


def raw(variant):
    r=dict(log_a=.2,log_b=-.4,t0=.7,t1=-.3,sigma_raw=.4)
    if variant=='P1_independent':r.pop('t1')
    if variant=='M1':r.pop('sigma_raw')
    return r


@pytest.mark.parametrize('variant',['P1_independent','P1_linked'])
def test_lookup_matches_native_posterior_and_future_count_states(variant):
    a=bank();p=infer(tensors(raw(variant)),variant,a,31)
    table=CountPosterior(raw(variant),variant,31)
    actual=table.probabilities(p.statistics.count,np.arange(1,17)[None,:])
    np.testing.assert_allclose(actual,p.post,atol=2e-14)
    np.testing.assert_allclose(actual.sum(-1),1,atol=2e-14)
    # Relabel all observed candidates into the extended future label space.
    n=np.zeros((4,33),int);n[:,[3,16,17,28]]=p.statistics.count[:,15,:4]
    extended=table.probabilities(n,16)
    np.testing.assert_allclose(extended[:,[3,16,17,28,32]],p.post[:,15,[0,1,2,3,16]],atol=2e-14)


def test_sigma_zero_voi_matches_original_v2_monte_carlo():
    from pgm_asc.v2.model import Spec,posterior,transforms
    from pgm_asc.v2.voi import voi
    a=bank();a['x']=np.zeros((4,16,3));p=tensors(raw('M1'));spec=Spec.of('M1')
    old=posterior(p,spec,a,transforms(a,spec,[0]));expected=voi(p,spec,a,old,key='nested')
    actual,error=trace(a,raw('M1'),'M1',V2['voi'],key='nested')
    assert error<1e-12
    np.testing.assert_allclose(actual['voi'],expected,atol=2e-8)
    np.testing.assert_allclose(actual['confidence'],old.confidence,atol=6e-8)


@pytest.mark.parametrize('variant',['P1_independent','P1_linked'])
def test_stopping_trace_is_prefix_causal_label_free_and_chunk_invariant(variant):
    a=bank();cfg={**V2['voi'],'simulations':16};cut=6
    expected,_=trace(a,raw(variant),variant,cfg,nodes=15,key='test',chunk=1)
    changed={**a,'ids':a['ids'].copy(),'correct':~a['correct'],'valid':a['valid'].copy()}
    changed['ids'][:,cut:]=3;changed['valid'][:,cut:]=False
    actual,_=trace(changed,raw(variant),variant,cfg,nodes=15,key='test',chunk=3)
    for k in expected:np.testing.assert_array_equal(actual[k][:,:cut],expected[k][:,:cut])
    full,_=trace(a,raw(variant),variant,cfg,nodes=15,key='test',chunk=4)
    for k in expected:np.testing.assert_array_equal(full[k],expected[k])


def test_standard_family_minimum_cap_and_strict_voi_threshold():
    from pgm_asc.v2.model import Spec
    from pgm_asc.v2.run import arm_family
    tr=dict(chosen=np.zeros((2,16),int),confidence=np.ones((2,16)),other=np.zeros((2,16)),voi=np.full((2,15),.3))
    fam=arm_family(tr,Spec.of('M1'))
    assert np.all(fam[.3][0]==15) # equality does not stop
    tr['voi'][:]=0
    assert np.all(arm_family(tr,Spec.of('M1'))[.3][0]==1)


def test_standard_report_preserves_palette_and_all_plugin_methods():
    from pgm_asc.v2.report import overview_groups,SLOTS
    extra=[f'p1_method_{i}' for i in range(13)]
    groups=overview_groups(extra)
    assert all(len(g)<=len(SLOTS) for g in groups)
    assert [m for group in groups for m in group[4:]]==extra
    assert len(overview_groups([]))==1


def test_trace_plugin_frozen_source_resume_and_observation_checks(tmp_path,monkeypatch):
    from pgm_asc import config
    from pgm_asc.p1_stopping import run as runner
    from pgm_asc.p1_stopping import plugin
    from pgm_asc.utils import file_sha256,write_json
    artifact=tmp_path/'artifacts';results=tmp_path/'results';(artifact/'v2').mkdir(parents=True)
    monkeypatch.setattr(config,'ARTIFACTS',artifact);monkeypatch.setattr(config,'RESULTS',results)
    monkeypatch.setitem(V2,'voi',{**V2['voi'],'simulations':8})
    a=bank();path=artifact/'v2'/'toy_math.npz'
    np.savez_compressed(path,**{f'{s}.{k}':v for s in ('dev','test') for k,v in a.items()})
    source=results/'p1_objectives';write_json(source/'protocol.json',dict(code_sha256={}))
    study={**runner.CONFIG,'source_dir':str(source)};fits={}
    for v in study['variants']:
        for m in study['training_methods']:
            fits[f'{v}__{m}']=dict(objective=dict(kind=m),best=dict(params=raw(v),normal_nodes=15))
    write_json(source/'selection/toy_math.json',dict(bank_sha256=file_sha256(path),experiment_id='toy',selection_locked_at='0',
        fits=fits,selected={k:k for k in fits},study=dict(integration=dict(sigma_bound=3.,normal_nodes=[15,31,61]))))
    paths=runner.run(['toy'],['math'],workers=1,study=study)
    c=json.loads((results/'p1_stopping/cells/toy_math.json').read_text())
    assert c['locked_at']<=c['evaluation_observations_loaded_at']
    context=dict(model='toy',dataset='math',split='test')
    plugin.cached.cache_clear()
    for d in runner.definitions().values():
        thr=.95 if d['stop']=='confidence' else .01
        if d['stop']=='voi':thr=V2['lambdas'][3]
        st,ch=plugin.decision(a,thr,context,d)
        assert np.all((st>=1)&(st<=15))
        assert np.all(((a['ids']==ch[:,None]) & a['valid'] & (np.arange(16)[None]<=st[:,None])).any(1))
        bad={**a,'ids':a['ids'].copy()};bad['ids'][0,0]=2
        with pytest.raises(ValueError,match='observation order'):plugin.decision(bad,thr,context,d)
    def fail(*args,**kwargs):raise AssertionError('resume must reuse traces')
    monkeypatch.setattr(runner,'trace',fail)
    assert runner.run(['toy'],['math'],workers=1,study=study)==paths
