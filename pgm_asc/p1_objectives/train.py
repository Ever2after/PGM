"""P1 joint / posterior-prefix / mixed losses on a common labeled cohort.

No graph or inference change: the Normal effect is integrated once per question
by the existing hierarchical model. Prefix losses normalize its native evidence
over observed candidate hypotheses and OTHER at every t=1..16.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from pgm_asc.hierarchical.model import KEYS, Statistics, infer, tensors
from pgm_asc.objectives.train import data_loss, prefix_weights


class Counts:
    def __init__(self, arrays):
        self.stats = Statistics(arrays)
        self.target = torch.tensor(self.stats.targets(arrays).ravel())
        self.end_target = self.target.reshape(-1,16)[:,-1]
        self.owner = torch.tensor(self.stats.owner)
        self.weights = {w:torch.tensor(prefix_weights(w)) for w in ('uniform','v1')}

    def state_losses(self, params, variant, nodes, sigma_bound=3.):
        marginal = self.stats.marginal(params,variant,nodes,sigma_bound)
        maximum = torch.segment_reduce(marginal,'max',lengths=self.stats.lengths)
        sums = torch.segment_reduce((marginal-maximum[self.owner]).exp(),'sum',lengths=self.stats.lengths)
        normalizer = maximum+sums.log()
        nll = (normalizer-marginal[self.target]).reshape(self.stats.questions,16)
        return marginal,nll

    def losses(self, params, variant, nodes, weighting='uniform', sigma_bound=3., need_prefix=True):
        if not need_prefix:
            marginal=self.stats.marginal(params,variant,nodes,sigma_bound)
            return -marginal[self.end_target].mean()/16,None
        marginal,nll=self.state_losses(params,variant,nodes,sigma_bound)
        return -marginal[self.end_target].mean()/16,(nll*self.weights[weighting]).sum(1).mean()


def initial_values(variant,settings,seed,sigma_bound):
    rng=np.random.default_rng(seed)
    raw={k:float(rng.normal(0,settings['initialization_sd'])) if seed else 0.
         for k in ('log_a','log_b','t0','t1')}
    sd=settings['sigma_starts'][settings['restart_seeds'].index(seed)]
    raw['sigma_raw']=float(sigma_bound*np.arctanh(sd/sigma_bound))
    return {k:raw[k] for k in KEYS[variant]}


def fit_one(counts,variant,objective,settings,seed,nodes=61,sigma_bound=3.):
    torch.set_num_threads(1)
    initial=initial_values(variant,settings,seed,sigma_bound)
    params=tensors(initial,True)
    opt=torch.optim.LBFGS(list(params.values()),lr=1.,max_iter=settings['max_iter'],
        tolerance_grad=settings['tolerance_grad'],tolerance_change=settings['tolerance_change'],
        history_size=20,line_search_fn='strong_wolfe')
    calls,started=0,time.perf_counter()
    def components():
        j,c=counts.losses(params,variant,nodes,objective.weighting,sigma_bound,objective.kind!='joint')
        data=data_loss(j,c,objective)
        return j,c,data,data+settings['ridge']*sum(v.square() for v in params.values())
    def closure():
        nonlocal calls
        calls+=1
        opt.zero_grad()
        loss=components()[-1]
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite P1 objective')
        loss.backward()
        return loss
    opt.step(closure)
    opt.zero_grad()
    j,c,data,loss=components()
    loss.backward()
    iterations=opt.state[next(iter(params.values()))]['n_iter']
    return dict(params={k:float(v.detach()) for k,v in params.items()},initial=initial,restart_seed=seed,
        joint_nll_per_response=float(j.detach()),prefix_nll=float(c.detach()) if c is not None else None,
        data_loss=float(data.detach()),penalized_loss=float(loss.detach()),
        gradient_max_abs=max(float(v.grad.abs()) for v in params.values()),iterations=iterations,closure_calls=calls,
        reached_iteration_limit=iterations>=settings['max_iter'],seconds=time.perf_counter()-started,
        parameters=len(params),normal_nodes=nodes)


@torch.no_grad()
def dev_score(counts,raw,variant,nodes,prefixes,sigma_bound):
    _,nll=counts.state_losses(tensors(raw),variant,nodes,sigma_bound)
    return float(nll[:,np.array(prefixes)-1].mean())


@torch.no_grad()
def integration_check(raw,variant,objective,train_counts,dev_counts,dev,coarse,fine,prefixes,sigma_bound):
    params=tensors(raw)
    values,posts,nexts=[],[],[]
    for n in (coarse,fine):
        j,c=train_counts.losses(params,variant,n,objective.weighting,sigma_bound)
        values.append(dict(train_joint=float(j),train_prefix=float(c),
            train_data=float(data_loss(j,c,objective)),
            dev_unique_nll=dev_score(dev_counts,raw,variant,n,prefixes,sigma_bound)))
        p=infer(params,variant,dev,n,sigma_bound)
        posts.append(p.post[:,np.array(prefixes)-1])
        nexts.append(np.array([p.next_answer(i,t) for i in range(len(dev['ids'])) for t in prefixes if t<16]))
        del p
    errors={k:abs(values[0][k]-values[1][k]) for k in values[0]}
    errors['dev_post_max_abs']=float(np.max(np.abs(posts[0]-posts[1])))
    errors['dev_next_max_abs']=float(np.max(np.abs(nexts[0]-nexts[1])))
    return dict(coarse_nodes=coarse,fine_nodes=fine,coarse=values[0],fine=values[1],errors=errors,max_error=max(errors.values()))
