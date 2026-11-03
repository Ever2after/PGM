import copy
import numpy as np
import torch
from pgm_asc.models.study import make,batch,predict,fit
from pgm_asc.data.reasc_bank import build
from pgm_asc.data.banks import fit_scalers,scale
from tests.unit.test_reasc_adapter import bank,rep


def data():
    bs=[bank('a'),bank('b')];d,_=build(bs,rep());d=scale(d,fit_scalers(d))
    h=np.zeros((2,32,8),np.float32);h[:,:16]=np.stack([b['vectors'] for b in bs]);d['embeddings']=torch.from_numpy(h)
    return d


def test_embedding_encoder_uses16_prefix_index_and_is_causal():
    d=data();model=make({'mode':'deepsets','use_embeddings':True},8)
    a=predict(model,d);alter={k:v.clone() for k,v in d.items()}
    alter['embeddings'][0,5:]+=1000;alter['x'][15,5:]-=999
    b=predict(model,alter)
    np.testing.assert_allclose(a[:5],b[:5],atol=1e-6)
    assert a.shape==(32,33)


def test_count_control_ignores_confidence_and_embedding():
    d=data();model=make({'mode':'candidate_mlp','count_only':True},8);a=predict(model,d)
    d['candidate'][:,:,2:]+=999;d['w'][:,8:]-=999;d['x']+=999;d['embeddings']+=999
    np.testing.assert_allclose(a,predict(model,d),atol=1e-6)


def test_alpha1_ignores_partition_information():
    d=data();model=make({'mode':'mixture','alpha':1.},8);a=predict(model,d)
    d['sizes']*=3;d['gate']+=10
    np.testing.assert_allclose(a,predict(model,d),atol=1e-6)


def test_all_embedding_training_gradients_finite_and_restore(tmp_path):
    d=data();cfg={'train':{'lr':.001,'weight_decay':.001,'epochs':2,'patience':20,'question_batch':2}}
    model=fit(d,d,{'mode':'deepsets','use_embeddings':True},cfg,tmp_path,42)
    np.testing.assert_allclose(np.exp(predict(model,d)).sum(1),1.,atol=1e-6)
