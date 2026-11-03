"""Three-state neural models, temporal filtering, and forward prediction."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pgm_asc.data.scaling import scaled


def feasible(count):
    return torch.stack([torch.ones_like(count,dtype=torch.bool),count>=2,count>=1],-1)


def transitions(new,changed):
    shape=(*new.shape,3,3)
    allowed=torch.ones(shape,dtype=torch.bool,device=new.device)
    allowed[...,1:,0]=False
    allowed[...,0,1:]=new[...,None]
    allowed[...,1,2]=changed
    allowed[...,2,1]=changed
    return allowed


class StateModel(nn.Module):
    def __init__(self,kind,dim=43):
        super().__init__()
        if kind not in ['mlp','masked','unmasked']:raise ValueError(kind)
        self.kind=kind
        self.network=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,3 if kind=='mlp' else 12))

    def forward(self,x,new,changed,count):
        outputs=self.network(x)
        emission=outputs[:,:,:3]
        point=feasible(count)
        if self.kind=='mlp':
            return emission.masked_fill(~point,-1e9).log_softmax(-1)
        allowed=transitions(new,changed)
        result=[emission[:,0].masked_fill(~point[:,0],-1e9).log_softmax(-1)]
        for t in range(1,x.shape[1]):
            logits=outputs[:,t,3:].reshape(-1,3,3)
            if self.kind=='masked':logits=logits.masked_fill(~allowed[:,t],-1e9)
            log_t=logits.log_softmax(-1)
            predicted=torch.logsumexp(result[-1][:,:,None]+log_t,dim=1)
            current=(predicted+emission[:,t]).masked_fill(~point[:,t],-1e9)
            result.append(current-current.logsumexp(-1,keepdim=True))
        return torch.stack(result,1)


def predict(model,observed,params,indices=None):
    if indices is None:indices=np.arange(len(observed['x']))
    model.eval();outputs=[]
    with torch.no_grad():
        for batch in np.array_split(indices,max(1,int(np.ceil(len(indices)/256)))):
            tensors=[torch.from_numpy(scaled(observed['x'][batch],params))]
            tensors += [torch.from_numpy(observed[k][batch]) for k in ['new','changed','count']]
            outputs.append(model(*tensors).numpy())
    return np.concatenate(outputs)
