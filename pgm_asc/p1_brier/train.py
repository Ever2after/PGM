"""Change only the proper event score, retaining the exact native event model.

B = mean over questions and prefix/horizon pairs of
    ((p_repair-y_repair)^2 + (p_damage-y_damage)^2) / 2.
There is no stay-class square, class balancing, or direct gain regression.
"""
from __future__ import annotations

import torch

from pgm_asc.p1_predictive.train import segment_lse
from pgm_asc.p1_transitions.train import Counts as NativeCounts
from pgm_asc.p1_transitions import train as native


class Counts(NativeCounts):
    def components(self,params,variant,nodes,weighting='uniform',sigma_bound=3.):
        if self.truth is None or self.target is None:raise ValueError('Training targets required')
        evidence=self.evidence(params,variant,nodes,sigma_bound)
        marginal=evidence[self.parent]+self.sf
        normalizer=segment_lse(marginal,self.stats.lengths,self.owner)
        nll=(normalizer-marginal[self.target]).reshape(-1,16)
        joint=-marginal[self.target.reshape(-1,16)[:,-1]].mean()/16
        prefix=(nll*self.weights[weighting]).sum(1).mean()
        probabilities,_=self.event_values(evidence,marginal,normalizer)
        pred=probabilities[self.context_inverse,:2]
        truth=torch.stack([self.truth==0,self.truth==1],1).to(pred.dtype)
        brier=(pred-truth).square().mean()
        return joint,prefix,brier,nll


def data_loss(joint,prefix,brier,mixing,beta):
    return native.data_loss(joint,prefix,brier,mixing,beta)


def fit_one(counts,variant,settings,seed,mixing,beta,weighting,nodes,sigma_bound=3.):
    result=native.fit_one(counts,variant,settings,seed,mixing,beta,weighting,nodes,sigma_bound)
    result['transition_brier']=result.pop('transition_nll')
    return result


@torch.no_grad()
def dev_score(counts,raw,variant,nodes,prefixes,sigma_bound):
    result=native.dev_score(counts,raw,variant,nodes,prefixes,sigma_bound)
    result['dev_transition_brier']=result.pop('dev_transition_nll')
    return result


@torch.no_grad()
def integration_check(raw,variant,train,dev,dev_full,coarse,fine,prefixes,weighting,mixing,beta,sigma_bound):
    result=native.integration_check(raw,variant,train,dev,dev_full,coarse,fine,prefixes,weighting,mixing,beta,sigma_bound)
    for key in ('coarse','fine','errors'):
        values=result[key]
        values['train_brier']=values.pop('train_transition')
        values['dev_transition_brier']=values.pop('dev_transition_nll')
    return result
