"""Differentiable native one-step predictive NLL with no gold conditioning.

For every active c hypothesis, unnormalized Q/A/B moments are evaluations of
sufficient states with increments (1,0,0,0), (0,1,1,0), (0,1,1,1). They encode
q, (1-q)/(m+theta), (1-q)*theta/(m+theta), respectively. This is the exact native
predictive generator, including the OTHER correct + new-wrong mixture. It is
directly evaluates predictive moments. Valid transitions also match consecutive
marginal-evidence ratios with consistent OTHER branching; invalid transitions
retain the native generator's zero probability rather than the legacy likelihood's
(1-q) observation factor.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from pgm_asc.diagnostics.metrics import EPS
from pgm_asc.hierarchical.model import Statistics,infer,latent_grid,tensors
from pgm_asc.objectives.train import prefix_weights
from pgm_asc.p1_objectives.train import initial_values
from pgm_asc.v2.model import log_prior


def segment_lse(values,lengths,owner):
    maximum=torch.segment_reduce(values,'max',lengths=lengths)
    total=torch.segment_reduce((values-maximum[owner]).exp(),'sum',lengths=lengths)
    return maximum+total.log()


class Counts:
    def __init__(self,arrays):
        self.stats=Statistics(arrays)
        st=self.stats
        self.target=torch.tensor(st.targets(arrays).ravel()) if 'correct' in arrays else None
        self.owner=torch.tensor(st.owner)
        self.weights={w:torch.tensor(prefix_weights(w)) for w in ('uniform','v1')}
        rows=np.flatnonzero(st.owner%16<15)
        needed,needed_inverse=np.unique(st.inverse[rows],return_inverse=True)
        parents=st.states[needed]
        self.states,inv=np.unique(np.vstack([st.states,parents+[1,0,0,0],
            parents+[0,1,1,0],parents+[0,1,1,1]]),axis=0,return_inverse=True)
        n,m=len(st.states),len(needed)
        self.parent=torch.tensor(inv[:n][st.inverse])
        self.q_child=torch.tensor(inv[n:n+m][needed_inverse])
        self.a_child=torch.tensor(inv[n+m:n+2*m][needed_inverse])
        self.b_child=torch.tensor(inv[n+2*m:][needed_inverse])
        self.pairs,pair_inv=np.unique(self.states[:,2:],axis=0,return_inverse=True)
        self.pair_inv=torch.tensor(pair_inv)
        self.nc=torch.tensor(self.states[:,0,None])
        self.wb=torch.tensor(self.states[:,1,None])
        self.pair_k=torch.tensor(self.pairs[:,1,None,None])
        self.sf=torch.tensor(st.sf)
        self.pred_sf=torch.tensor(st.sf[rows])
        pred_owner=(st.owner[rows]//16)*15+st.owner[rows]%16
        self.pred_owner=torch.tensor(pred_owner)
        self.pred_lengths=torch.tensor(st.allowed[:,:15].sum(2).ravel())
        ids=np.asarray(arrays['ids'],int)
        target_count=np.take_along_axis(st.count[:,:15,:16],ids[:,1:,None],2)[...,0].ravel()
        self.new=torch.tensor(target_count[pred_owner]==0)
        self.other=torch.tensor(st.hyp[rows]==16)
        self.matches=torch.tensor(st.hyp[rows]==ids[:,1:].ravel()[pred_owner])
        self.log_count=torch.tensor(np.log(np.maximum(target_count[pred_owner],1)))
        self.valid_next=torch.tensor(np.asarray(arrays['valid'][:,1:],bool))

    def evidence(self,params,variant,nodes,sigma_bound=3.):
        grid,lt,lp,rising=latent_grid(params,variant,nodes,sigma_bound)
        prior=log_prior(params,grid)
        normal=lp-prior[:,None]
        pair=torch.logsumexp(self.pair_k*lt[None]-rising[self.pairs[:,0]]+normal[None],-1)
        return torch.logsumexp(self.nc*grid.lq+self.wb*grid.l1q+pair[self.pair_inv]+prior,-1)

    def state_values(self,params,variant,nodes,sigma_bound=3.):
        evidence=self.evidence(params,variant,nodes,sigma_bound)
        marginal=evidence[self.parent]+self.sf
        normalizer=segment_lse(marginal,self.stats.lengths,self.owner)
        q,a,b=evidence[self.q_child],evidence[self.a_child],evidence[self.b_child]
        new=torch.where(self.other,torch.logaddexp(q,b),b)
        repeat=torch.where(self.matches,q,a+self.log_count)
        contribution=torch.where(self.new,new,repeat)+self.pred_sf
        numerator=segment_lse(contribution,self.pred_lengths,self.pred_owner)
        log_predictive=(numerator-normalizer.reshape(-1,16)[:,:15].ravel()).reshape(-1,15)
        # Invalid next answers have native probability zero. Clipping is constant
        # and matches the unchanged evaluator; it adds no invalid-answer model.
        log_predictive=torch.where(self.valid_next,log_predictive,torch.full_like(log_predictive,np.log(EPS)))
        log_predictive=log_predictive.clamp(min=np.log(EPS),max=0.)
        return marginal,normalizer,log_predictive

    def components(self,params,variant,nodes,weighting='uniform',sigma_bound=3.):
        if self.target is None:raise ValueError('Joint/prefix components require training labels')
        marginal,normalizer,logp=self.state_values(params,variant,nodes,sigma_bound)
        nll=(normalizer-marginal[self.target]).reshape(-1,16)
        j=-marginal[self.target.reshape(-1,16)[:,-1]].mean()/16
        c=(nll*self.weights[weighting]).sum(1).mean()
        return j,c,-logp.mean(),nll


def data_loss(joint,prefix,predictive,mixing,gamma):
    mixed=(joint+mixing*prefix)/(1+mixing)
    return (mixed+gamma*predictive)/(1+gamma)


def fit_one(counts,variant,settings,seed,mixing,gamma,weighting,nodes,sigma_bound=3.):
    torch.set_num_threads(1)
    initial=initial_values(variant,settings,seed,sigma_bound)
    params=tensors(initial,True)
    opt=torch.optim.LBFGS(list(params.values()),lr=1.,max_iter=settings['max_iter'],
        tolerance_grad=settings['tolerance_grad'],tolerance_change=settings['tolerance_change'],
        history_size=20,line_search_fn='strong_wolfe')
    calls,started=0,time.perf_counter()
    def components():
        j,c,n,_=counts.components(params,variant,nodes,weighting,sigma_bound)
        data=data_loss(j,c,n,mixing,gamma)
        return j,c,n,data,data+settings['ridge']*sum(p.square() for p in params.values())
    def closure():
        nonlocal calls
        calls+=1;opt.zero_grad()
        loss=components()[-1]
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite predictive objective')
        loss.backward();return loss
    opt.step(closure);opt.zero_grad()
    j,c,n,data,loss=components();loss.backward()
    iterations=opt.state[next(iter(params.values()))]['n_iter']
    return dict(params={k:float(v.detach()) for k,v in params.items()},initial=initial,restart_seed=seed,
        joint_nll_per_response=float(j.detach()),prefix_nll=float(c.detach()),predictive_nll=float(n.detach()),
        data_loss=float(data.detach()),penalized_loss=float(loss.detach()),
        gradient_max_abs=max(float(v.grad.abs()) for v in params.values()),iterations=iterations,closure_calls=calls,
        reached_iteration_limit=iterations>=settings['max_iter'],seconds=time.perf_counter()-started,
        parameters=len(params),normal_nodes=nodes,mixing=mixing,gamma=gamma,weighting=weighting)


@torch.no_grad()
def dev_score(counts,raw,variant,nodes,prefixes,sigma_bound):
    _,_,n,nll=counts.components(tensors(raw),variant,nodes,'uniform',sigma_bound)
    return dict(dev_unique_nll=float(nll[:,np.array(prefixes)-1].mean()),dev_predictive_nll=float(n))


@torch.no_grad()
def integration_check(raw,variant,train,dev,dev_full,coarse,fine,prefixes,weighting,mixing,gamma,sigma_bound):
    values,posts,nexts=[],[],[]
    params=tensors(raw)
    for nodes in (coarse,fine):
        j,c,n,_=train.components(params,variant,nodes,weighting,sigma_bound)
        values.append(dict(train_joint=float(j),train_prefix=float(c),train_predictive=float(n),
            train_data=float(data_loss(j,c,n,mixing,gamma)),**dev_score(dev,raw,variant,nodes,prefixes,sigma_bound)))
        p=infer(params,variant,dev_full,nodes,sigma_bound)
        posts.append(p.post[:,np.array(prefixes)-1])
        nexts.append(np.array([p.next_answer(i,t) for i in range(len(dev_full['ids'])) for t in prefixes if t<16]));del p
    errors={k:abs(values[0][k]-values[1][k]) for k in values[0]}
    errors['dev_post_max_abs']=float(np.max(np.abs(posts[0]-posts[1])))
    errors['dev_next_max_abs']=float(np.max(np.abs(nexts[0]-nexts[1])))
    return dict(coarse_nodes=coarse,fine_nodes=fine,coarse=values[0],fine=values[1],errors=errors,max_error=max(errors.values()))
