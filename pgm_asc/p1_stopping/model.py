"""P1 VOI with exact count-posterior updates and the v2 Monte Carlo rule.

Sufficient-state lookup replaces repeated latent integration, not the model.
Every continuation holds (c,q,theta) fixed and updates its posterior at each
lookahead horizon. This is the stopping VOI, rather than the fixed-decision
repair/damage diagnostic used in the earlier P1 studies.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import gammaln, logsumexp

from pgm_asc.hierarchical.model import infer, latent_grid, tensors


class CountPosterior:
    @torch.no_grad()
    def __init__(self, raw, variant, nodes=61, sigma_bound=3.):
        p=tensors(raw)
        grid,lt,lp,rising=latent_grid(p,variant,nodes,sigma_bound)
        # Normal integral for each possible (wrong total, wrong group count).
        pairs=[(m,k) for m in range(17) for k in (range(1,m+1) if m else [0])]
        pair={}
        from pgm_asc.v2.model import log_prior
        normal_lp=lp-log_prior(p,grid)[:,None]
        for m,k in pairs:
            pair[m,k]=torch.logsumexp(k*lt-rising[m]+normal_lp,-1).numpy()
        self.evidence=np.full((17,17,17,17),np.nan)
        prior=log_prior(p,grid).numpy()
        for t in range(17):
            for valid in range(t+1):
                for nc in range(valid+1):
                    m=valid-nc
                    for k in (range(1,m+1) if m else [0]):
                        self.evidence[t,valid,nc,k]=logsumexp(nc*grid.lq.numpy()+(t-nc)*grid.l1q.numpy()+pair[m,k]+prior)
        self.log_factorial=np.r_[0.,gammaln(np.arange(1,17))]

    def probabilities(self, counts, total):
        """Counts [..., L], last label is the single unseen-correct hypothesis."""
        n=np.asarray(counts,dtype=int)
        valid=n.sum(-1)
        seen=n>0
        k=seen.sum(-1,keepdims=True)-seen
        sf=self.log_factorial[n]
        total=np.asarray(total)
        if total.ndim:total=np.broadcast_to(total,valid.shape)[...,None]
        logits=self.evidence[total,valid[...,None],n,k]+sf.sum(-1,keepdims=True)-sf
        allowed=seen.copy();allowed[...,-1]=True
        logits=np.where(allowed,logits,-np.inf)
        if np.isnan(logits).any():raise ValueError('Infeasible sufficient state')
        return np.exp(logits-logsumexp(logits,-1,keepdims=True))


def decisions(counts,post):
    seen=counts[...,:-1]>0
    tied=seen & (counts[...,:-1]==counts[...,:-1].max(-1,keepdims=True))
    chosen=np.where(seen.any(-1),np.where(tied,post[...,:-1],-1.).argmax(-1),-1)
    conf=np.where(chosen>=0,np.take_along_axis(post,np.maximum(chosen,0)[...,None],-1)[...,0],0.)
    return chosen,conf


def continuation_values(counts0,total,c,qs,theta,uniforms,table,marks):
    """Batched MC, with posterior tie-break and exact posterior confidence."""
    # [questions, simulations, labels]. 16=new correct, 17..31=new wrong,
    # 32=the remaining unseen-correct hypothesis, exactly as the v2 simulator.
    nq,sims=c.shape
    counts=np.zeros((nq,sims,33),int);counts[:,:,:16]=counts0[:,None,:]
    nxt=np.full((nq,sims),17)
    qi,si=np.indices((nq,sims));out={}
    for step in range(1,uniforms.shape[1]+1):
        u=uniforms[:,step-1]
        correct=u[:,0]<qs
        wrong=counts.copy();wrong[qi,si,c]=0
        m=wrong.sum(-1)
        new=u[:,1]<theta/(m+theta)
        cum=np.cumsum(wrong,-1)/np.maximum(m,1)[...,None]
        old=(u[:,2,:,None]<cum).argmax(-1)
        label=np.where(correct,c,np.where(new|(m==0),nxt,old))
        nxt+=~correct & (new|(m==0))
        counts[qi,si,label]+=1
        if step in marks:
            post=table.probabilities(counts,total+step)
            _,conf=decisions(counts,post)
            out[step]=conf.mean(1)
    return out


def trace(arrays,raw,variant,cfg,nodes=61,sigma_bound=3.,key='',chunk=64):
    """Labels/features/tokens are not read; every statistic is prefix causal."""
    obs={k:arrays[k] for k in ('ids','valid','question_id')}
    p=infer(tensors(raw),variant,obs,nodes,sigma_bound)
    table=CountPosterior(raw,variant,nodes,sigma_bound)
    counts=p.statistics.count
    post=table.probabilities(counts,np.arange(1,17)[None,:])
    post_error=float(np.max(np.abs(post-p.post)))
    if post_error>1e-10:raise ValueError('Count lookup differs from native P1 posterior')
    chosen,confidence=decisions(counts,post)
    cap=16;sims=cfg['simulations'];voi=np.zeros((len(counts),15))
    # Identical base seed and stream layout to v2. A shared key pairs the P1 arms.
    rngs=[np.random.default_rng([cfg['seed'],i,sum(map(ord,key))]) for i in range(len(counts))]
    for start in range(0,len(counts),chunk):
        end=min(start+chunk,len(counts));rows=range(start,end)
        for t in range(1,cap):
            hmax=cap-t;marks=sorted({h for h in cfg['horizons'] if h<hmax}|{hmax})
            c=[];qs=[];theta=[];uniforms=[]
            for i in rows:
                latent=p.draw(i,t,rngs[i].random(sims))
                c.append(latent[0]);qs.append(latent[1]);theta.append(latent[2])
                uniforms.append(rngs[i].random((hmax,3,sims)))
            future=continuation_values(counts[start:end,t-1,:16],t,np.array(c),np.array(qs),np.array(theta),np.array(uniforms),table,marks)
            value=np.stack([(future[h]-confidence[start:end,t-1])/h for h in marks])
            voi[start:end,t-1]=np.maximum(0.,value.max(0))
    return dict(chosen=chosen.astype(np.int8),confidence=confidence.astype(np.float32),
        other=post[:,:,16].astype(np.float32),voi=voi.astype(np.float32)),post_error
