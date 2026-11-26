"""Direct proper transition score with exact finite-horizon CRP kernels.

All event probabilities marginalize c,q,theta given past observations only.
Gold current/future correctness enters the three-class training target only.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from pgm_asc.diagnostics import metrics
from pgm_asc.p1_predictive.train import Counts as PredictiveCounts,segment_lse
from pgm_asc.p1_objectives.train import initial_values
from pgm_asc.hierarchical.model import infer,tensors
from pgm_asc.p1_transitions.kernel import hypothesis_signature,success_terms


class Counts(PredictiveCounts):
    def __init__(self,arrays,prefixes=(1,2,4,8),horizons=(1,2,4,8)):
        super().__init__(arrays)
        st=self.stats
        self.pairs_th=[(t,h) for t in prefixes for h in horizons if t+h<=16]
        if not self.pairs_th:raise ValueError('No feasible transition pairs')
        row_map=np.full((st.questions*16,17),-1,int)
        row_map[st.owner,st.hyp]=np.arange(len(st.owner))
        contexts={};inverse=[];representatives=[]
        for qi in range(st.questions):
            for t,h in self.pairs_th:
                count=st.count[qi,t-1,:16]
                observed=[]
                for j in range(t):
                    lab=int(arrays['ids'][qi,j])
                    if arrays['valid'][qi,j] and lab not in observed:observed.append(lab)
                key=(t,tuple(int(count[k]) for k in observed),h)
                if key not in contexts:
                    contexts[key]=len(contexts);representatives.append((qi,t,h))
                inverse.append(contexts[key])
        self.context_inverse=torch.tensor(inverse)
        self.contexts=len(contexts)
        buckets={};current_rows=[];current_valid=[];damage_possible=[];owner_refs=[]
        for ci,(qi,t,h) in enumerate(representatives):
            counts=st.count[qi,t-1,:16]
            first=np.full(16,1000,int)
            for j in range(t):
                if arrays['valid'][qi,j]:
                    lab=int(arrays['ids'][qi,j]);first[lab]=min(first[lab],j-t)
            leader=int(np.where(counts==counts.max(),first,1000).argmin()) if counts.sum() else -1
            owner=qi*16+t-1;owner_refs.append(owner)
            current_valid.append(leader>=0)
            current_rows.append(row_map[owner,leader] if leader>=0 else 0)
            can_damage=False
            if leader>=0:
                nc=counts[leader]
                can_damage=h>nc or any(k!=leader and n>0 and
                    (n+h>nc or (n+h==nc and first[k]<first[leader])) for k,n in enumerate(counts))
            damage_possible.append(can_damage)
            for hyp in np.flatnonzero(st.allowed[qi,t-1]):
                row=row_map[owner,hyp];base=st.states[st.inverse[row]]
                nc,early,late,unseen=hypothesis_signature(counts,first,int(hyp))
                for correct,wrong,groups,coefficient in success_terms(nc,early,late,h,unseen):
                    state=tuple((base+np.array([correct,wrong,wrong,groups])).tolist())
                    bucket=2*ci+int(hyp==leader)
                    # Factorials never exceed the 16-response bank's range.
                    key=(bucket,state)
                    buckets[key]=buckets.get(key,0.)+coefficient*np.exp(st.sf[row])
        term_states=np.array([key[1] for key in buckets],int).reshape(-1,4)
        original=len(self.states)
        states,inv=np.unique(np.vstack([self.states,term_states]),axis=0,return_inverse=True)
        remap=inv[:original]
        for attr in ('parent','q_child','a_child','b_child'):
            setattr(self,attr,torch.tensor(remap[getattr(self,attr).numpy()]))
        self.states=states
        self.pairs,pair_inv=np.unique(states[:,2:],axis=0,return_inverse=True)
        self.pair_inv=torch.tensor(pair_inv);self.nc=torch.tensor(states[:,0,None]);self.wb=torch.tensor(states[:,1,None])
        self.pair_k=torch.tensor(self.pairs[:,1,None,None])
        self.term_state=torch.tensor(inv[original:])
        self.term_bucket=torch.tensor([key[0] for key in buckets])
        self.term_owner=torch.tensor([owner_refs[key[0]//2] for key in buckets])
        self.term_log_weight=torch.tensor(np.log(list(buckets.values())))
        self.current_row=torch.tensor(current_rows);self.owner_ref=torch.tensor(owner_refs)
        self.current_valid=torch.tensor(current_valid);self.damage_possible=torch.tensor(damage_possible)
        self.truth=None
        if 'correct' in arrays:
            hit=metrics.labels(arrays,metrics.reference(arrays))
            truth=[]
            for qi in range(st.questions):
                for t,h in self.pairs_th:
                    now,future=hit[qi,t-1],hit[qi,t+h-1]
                    truth.append(0 if not now and future else 1 if now and not future else 2)
            self.truth=torch.tensor(truth)

    def event_values(self,evidence,marginal,normalizer):
        mass=(evidence[self.term_state]+self.term_log_weight-normalizer[self.term_owner]).exp()
        success=torch.zeros(self.contexts*2).scatter_add(0,self.term_bucket,mass).reshape(-1,2)
        current=torch.where(self.current_valid,(marginal[self.current_row]-normalizer[self.owner_ref]).exp(),0.)
        repair=success[:,0]
        damage=torch.where(self.damage_possible,current-success[:,1],0.).clamp(min=0.,max=1.)
        repair=repair.clamp(min=0.,max=1.)
        stay=(1-repair-damage).clamp(min=0.,max=1.)
        return torch.stack([repair,damage,stay],1),current

    def event_probabilities(self,params,variant,nodes,sigma_bound=3.):
        evidence=self.evidence(params,variant,nodes,sigma_bound)
        marginal=evidence[self.parent]+self.sf
        normalizer=segment_lse(marginal,self.stats.lengths,self.owner)
        probabilities,current=self.event_values(evidence,marginal,normalizer)
        return probabilities[self.context_inverse].reshape(self.stats.questions,-1,3),current[self.context_inverse].reshape(self.stats.questions,-1)

    def components(self,params,variant,nodes,weighting='uniform',sigma_bound=3.):
        if self.truth is None or self.target is None:raise ValueError('Training targets required')
        evidence=self.evidence(params,variant,nodes,sigma_bound)
        marginal=evidence[self.parent]+self.sf
        normalizer=segment_lse(marginal,self.stats.lengths,self.owner)
        nll=(normalizer-marginal[self.target]).reshape(-1,16)
        joint=-marginal[self.target.reshape(-1,16)[:,-1]].mean()/16
        prefix=(nll*self.weights[weighting]).sum(1).mean()
        probabilities,_=self.event_values(evidence,marginal,normalizer)
        p=probabilities[self.context_inverse, self.truth]
        transition=-p.clamp(min=metrics.EPS,max=1.).log().mean()
        return joint,prefix,transition,nll


def data_loss(joint,prefix,transition,mixing,beta):
    mixed=(joint+mixing*prefix)/(1+mixing)
    return (mixed+beta*transition)/(1+beta)


def fit_one(counts,variant,settings,seed,mixing,beta,weighting,nodes,sigma_bound=3.):
    torch.set_num_threads(1)
    initial=initial_values(variant,settings,seed,sigma_bound);params=tensors(initial,True)
    opt=torch.optim.LBFGS(list(params.values()),lr=1.,max_iter=settings['max_iter'],
        tolerance_grad=settings['tolerance_grad'],tolerance_change=settings['tolerance_change'],
        history_size=20,line_search_fn='strong_wolfe')
    calls,started=0,time.perf_counter()
    def components():
        j,c,t,_=counts.components(params,variant,nodes,weighting,sigma_bound)
        data=data_loss(j,c,t,mixing,beta)
        return j,c,t,data,data+settings['ridge']*sum(p.square() for p in params.values())
    def closure():
        nonlocal calls
        calls+=1;opt.zero_grad();loss=components()[-1]
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite direct-transition objective')
        loss.backward();return loss
    opt.step(closure);opt.zero_grad();j,c,t,data,loss=components();loss.backward()
    iterations=opt.state[next(iter(params.values()))]['n_iter']
    return dict(params={k:float(p.detach()) for k,p in params.items()},initial=initial,restart_seed=seed,
        joint_nll_per_response=float(j.detach()),prefix_nll=float(c.detach()),transition_nll=float(t.detach()),
        data_loss=float(data.detach()),penalized_loss=float(loss.detach()),gradient_max_abs=max(float(p.grad.abs()) for p in params.values()),
        iterations=iterations,closure_calls=calls,reached_iteration_limit=iterations>=settings['max_iter'],
        seconds=time.perf_counter()-started,parameters=len(params),normal_nodes=nodes,mixing=mixing,beta=beta,weighting=weighting)


@torch.no_grad()
def dev_score(counts,raw,variant,nodes,prefixes,sigma_bound):
    _,_,transition,nll=counts.components(tensors(raw),variant,nodes,'uniform',sigma_bound)
    return dict(dev_unique_nll=float(nll[:,np.array(prefixes)-1].mean()),dev_transition_nll=float(transition))


@torch.no_grad()
def integration_check(raw,variant,train,dev,dev_full,coarse,fine,prefixes,weighting,mixing,beta,sigma_bound):
    values,posts,nexts,events=[],[],[],[];params=tensors(raw)
    for nodes in (coarse,fine):
        j,c,t,_=train.components(params,variant,nodes,weighting,sigma_bound)
        values.append(dict(train_joint=float(j),train_prefix=float(c),train_transition=float(t),
            train_data=float(data_loss(j,c,t,mixing,beta)),**dev_score(dev,raw,variant,nodes,prefixes,sigma_bound)))
        probabilities,_=dev.event_probabilities(params,variant,nodes,sigma_bound);events.append(probabilities.numpy())
        p=infer(params,variant,dev_full,nodes,sigma_bound)
        posts.append(p.post[:,np.array(prefixes)-1]);nexts.append(np.array([p.next_answer(i,t) for i in range(len(dev_full['ids'])) for t in prefixes if t<16]));del p
    errors={k:abs(values[0][k]-values[1][k]) for k in values[0]}
    errors['dev_event_max_abs']=float(np.max(np.abs(events[0]-events[1])))
    errors['dev_post_max_abs']=float(np.max(np.abs(posts[0]-posts[1])))
    errors['dev_next_max_abs']=float(np.max(np.abs(nexts[0]-nexts[1])))
    return dict(coarse_nodes=coarse,fine_nodes=fine,coarse=values[0],fine=values[1],errors=errors,max_error=max(errors.values()))
