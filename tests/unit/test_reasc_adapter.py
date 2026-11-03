import copy
import numpy as np
import torch
from pgm_asc.data.reasc_bank import pack,build,traces
from pgm_asc.data.banks import INPUT_KEYS,fit_scalers,scale
from pgm_asc.experiments.reasc_comparison import prefix_mass,fit
from pgm_asc.models.selector import PGM,predict
from pgm_asc.policies.continuation import continuation_features
from pgm_asc.collection.reasc_judge import verdict,literal


def bank(qid='fake'):
    r=[]
    for i in range(16):
        r.append({'prediction':['1/2',r'\frac{1}{2}',''][i%3],
                  'reasoning_mean_logprob':-.2,'answer_mean_logprob':-.4,'reasoning_mean_entropy':2.,
                  'generated_token_count':10+i,'reasoning_token_count':8,'answer_token_count':1,
                  'truncated':False,'reference_chat_tokens':20+i})
    return {'question':{'question_id':qid},'records':r,'vectors':np.random.default_rng(7).normal(size=(16,8)),
            'labels':{'canonical_correct':[True,True,False]}}


def rep():
    return {'mean':[0.]*8,'thresholds':[.1,.4,.7,.9]}


def test_capacity_length_and_exact_groups():
    b=bank();data,meta=build([b],rep());lp,_=predict(PGM(32),data);h=traces(lp,data)
    assert data['x'].shape==(16,32,9) and data['w'].shape==(16,25)
    assert len(h)==1 and len(h[0]['chosen'])==16
    assert meta[-1]['representatives']==['1/2',r'\frac{1}{2}','']
    assert data['gold_groups'][-1,:3].tolist()==[True,True,False]
    assert np.isclose(prefix_mass().sum(),1.)
    assert continuation_features(h,data).shape==(16,31)
    np.testing.assert_allclose(np.exp(lp).sum(1),1.,atol=1e-6)


def test_packing_and_prediction_future_and_label_invariance():
    b=bank();a=pack(b['records'],b['vectors'],rep());altered=copy.deepcopy(b)
    for r in altered['records'][5:]:r.update(prediction='changed',reasoning_mean_entropy=99.)
    altered['vectors'][5:]=99.;altered['labels']['canonical_correct']=[False]*4
    z=pack(altered['records'],altered['vectors'],rep())
    for t in range(5):
        for key in INPUT_KEYS:np.testing.assert_array_equal(a[t][key],z[t][key])
    d,_=build([b],rep());e,_=build([altered],rep());model=PGM(32)
    pa,_=predict(model,d);pb,_=predict(model,e)
    np.testing.assert_array_equal(pa[:5],pb[:5])


def test_training_smoke_handles_sixteen_not_thirtytwo(tmp_path):
    b=bank();d,_=build([b,bank('second')],rep());d=scale(d,fit_scalers(d))
    config={'train':{'lr':.001,'weight_decay':.001,'question_batch':2,'epochs':2,'patience':20}}
    m=fit(d,d,{'mode':'mixture'},config,tmp_path,epochs=2)
    lp,_=predict(m,d)
    assert lp.shape==(32,33) and (tmp_path/'checkpoint.pt').exists()


def test_judge_preserves_reference_template_and_strict_verdict():
    import pytest
    assert verdict('[Correct]') and not verdict('Incorrect')
    with pytest.raises(ValueError):verdict('Perhaps correct')
    template=literal('third_party/xverify_official/src/xVerify/prompts/base_template.py','BASE_TEMPLATE')['xVerify-3B-Ia']
    assert r'\n' in template and '{query}' in template


def test_metric_separates_prefix_and_esc_full_bank_verdicts():
    from pgm_asc.evaluation.reasc_comparison import row,paired
    b=bank();b['labels']['raw_response_correct']=[False]*4+[True]*12
    for r in b['records']:
        r.update(feature_input_tokens=30,feature_seconds_share=.1,native_generation_seconds_share=.2,
                 extraction_inline={'input_tokens':20,'output_tokens':3,'seconds_share':.1},
                 extraction_final={'input_tokens':20,'output_tokens':3,'seconds_share':.1})
    early=row(b,3,0,'confidence',False);esc=row(b,3,0,'esc_unanimous_block',False)
    assert not early['reference_correct'] and esc['reference_correct']
    pgm=row(b,3,0,'confidence',True)
    assert pgm['reference_proxy_T']==early['reference_proxy_T']
    assert pgm['dense_2N_approx_T']>early['dense_2N_approx_T']
    p=paired([pgm],[early])
    assert p['accuracy_difference_ci95']==[0.,0.] and p['reference_cost_ratio']==1.
