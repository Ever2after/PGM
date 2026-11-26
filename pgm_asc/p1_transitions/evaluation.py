"""Exact-event sensitivity alongside the unchanged paired-MC evaluation."""
from __future__ import annotations

import numpy as np
import torch
from pgm_asc.diagnostics import metrics


def transition_nll(future,arrays,prefixes,horizons):
    hit=metrics.labels(arrays,metrics.reference(arrays));values=[]
    for pi,t in enumerate(prefixes):
        for hi,h in enumerate(horizons):
            if t+h>16:continue
            r,d=future['repair'][:,pi,hi],future['damage'][:,pi,hi]
            truth=np.where(~hit[:,t-1]&hit[:,t+h-1],0,np.where(hit[:,t-1]&~hit[:,t+h-1],1,2))
            p=np.stack([r,d,1-r-d],1)[np.arange(len(hit)),truth]
            values.append(-np.log(np.clip(p,metrics.EPS,1.)))
    return np.mean(values,axis=0)


@torch.no_grad()
def exact_forecast(counts,params,variant,nodes,bound,prefixes,horizons,mc_future):
    event,current=counts.event_probabilities(params,variant,nodes,bound)
    event,current=event.numpy(),current.numpy()
    if np.min(event)<-1e-9 or np.max(np.abs(event.sum(2)-1))>1e-9:
        raise ValueError('Exact event probabilities are not normalized')
    result={k:v.copy() for k,v in mc_future.items()}
    result.update({k:np.full_like(mc_future['gain'],np.nan) for k in
        ('repair','damage','gain','future_reference_correct','gain_mc_se','repair_mc_se','damage_mc_se','current_correct_mc')})
    for ei,(t,h) in enumerate(counts.pairs_th):
        pi,hi=prefixes.index(t),horizons.index(h)
        r,d=event[:,ei,0],event[:,ei,1]
        result['repair'][:,pi,hi]=r;result['damage'][:,pi,hi]=d;result['gain'][:,pi,hi]=r-d
        result['current_correct_mc'][:,pi,hi]=current[:,ei]
        result['future_reference_correct'][:,pi,hi]=np.clip(current[:,ei]+r-d,0.,1.)
        for k in ('gain_mc_se','repair_mc_se','damage_mc_se'):result[k][:,pi,hi]=0.
    return result


def exact_score_keys(values):
    return {k:v for k,v in values.items() if k.startswith(('repair_','damage_','gain_','future_reference_correct_'))}
