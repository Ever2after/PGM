import copy
import numpy as np
import torch
import pytest

from pgm_asc.models.state import StateModel
from pgm_asc.models.state import transitions
from pgm_asc.models.state import feasible
from pgm_asc.data.state import observed_features


def inputs():
    torch.manual_seed(42)
    x=torch.randn(3,8,43)
    count=torch.tensor([[1,1,2,2,2,3,3,3],[0,1,1,1,2,2,2,2],[1,2,3,4,4,4,4,5]])
    new=count>torch.cat([torch.zeros((3,1),dtype=int),count[:,:-1]],1)
    changed=new.clone()
    return x,new,changed,count


@pytest.mark.parametrize('kind',['mlp','unmasked','masked'])
def test_state_predictions_are_causal_and_normalized(kind):
    x,new,changed,count=inputs();model=StateModel(kind).eval()
    a=model(x,new,changed,count)
    altered=x.clone();altered[:,4:]=10000
    nc=new.clone();nc[:,4:]=~nc[:,4:]
    b=model(altered,nc,changed,count)
    assert torch.allclose(a[:,:4],b[:,:4],atol=1e-6)
    assert torch.allclose(a.exp().sum(-1),torch.ones((3,8)),atol=1e-6)
    point=feasible(count)
    assert (a.exp()[~point]==0).all()


def test_transition_constraints_match_state_semantics():
    new=torch.tensor([False,True,False,True]);changed=torch.tensor([False,False,True,True])
    allowed=transitions(new,changed)
    assert not allowed[:,1:,0].any()
    assert torch.equal(allowed[:,0,1],new)
    assert torch.equal(allowed[:,1,2],changed)
    assert torch.equal(allowed[:,2,1],changed)
    assert allowed.diagonal(dim1=-2,dim2=-1).all()


@pytest.mark.parametrize('kind',['mlp','unmasked','masked'])
def test_state_loss_has_finite_gradients(kind):
    x,new,changed,count=inputs();model=StateModel(kind)
    lp=model(x,new,changed,count)
    # A valid U-only path is possible for every question.
    loss=-lp[:,:,0].mean();loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_observed_feature_function_is_invariant_to_labels_and_future():
    d={'candidate_mask':torch.tensor([[1,0,1]]*8,dtype=bool),
       'raw_w':torch.zeros(8,25),'latest_x':torch.zeros(8,9)}
    lp=np.full((1,8,9),-1e9);lp[:,:,0]=np.log(.8);lp[:,:,8]=np.log(.2)
    a=observed_features(d,lp)
    other=copy.deepcopy(d);other['target']=torch.ones(8);other['gold_groups']=torch.ones(8,33)
    altered=lp.copy();altered[:,5:,0]=np.log(.1);altered[:,5:,8]=np.log(.9)
    b=observed_features(other,altered)
    assert np.array_equal(a['x'][:,:5],b['x'][:,:5])
    assert a['x'].shape==(1,8,43)


def test_value_fit_excludes_held_question_labels():
    from pgm_asc.training.value import fit_bundle
    from pgm_asc.models.value import ridge_predict
    from pgm_asc.models.value import value_x
    rng=np.random.default_rng(42)
    obs={'x':rng.normal(size=(12,8,43)).astype(np.float32),'chosen':np.zeros((12,8),int)}
    lp=np.log(np.tile([.2,.1,.7],(12,8,1)))
    correct=rng.integers(0,2,size=(12,8));cost=np.cumsum(rng.integers(10,100,size=(12,8)),1)
    index=np.arange(9)
    a=fit_bundle(obs,lp,correct,cost,index,[3e-6])
    altered=correct.copy();altered[9:]=1-altered[9:]
    b=fit_bundle(obs,lp,altered,cost,index,[3e-6])
    x=value_x(obs,lp).reshape(-1,46)
    assert np.allclose(ridge_predict(a['fixed4'],x),ridge_predict(b['fixed4'],x))
    for t in range(1,7):
        assert np.allclose(ridge_predict(a['bellman']['3e-06']['models'][t],x),ridge_predict(b['bellman']['3e-06']['models'][t],x))


def test_value_inference_does_not_change_earlier_decisions_with_future_values():
    from pgm_asc.policies.stopping import stop_indices
    obs={'chosen':np.zeros((2,8),int)}
    lp=np.log(np.tile([.05,.05,.9],(2,8,1)))
    q=np.zeros((2,8,2));q[:,:,0]=.1;q[:,:,1]=100
    p={'family':'bellman','gamma':None,'lambda':3e-6}
    a,_=stop_indices(obs,lp,p,q)
    q[:,3:]=1e9
    b,_=stop_indices(obs,lp,p,q)
    assert np.array_equal(a,b) and (a==1).all()
