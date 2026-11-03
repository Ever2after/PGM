"""Causal stopping rules and calibration-only policy selection."""
from __future__ import annotations

import numpy as np


GAMMAS = [.5, .6, .7, .8, .85, .9, .925, .95, .975, .99, .995, None]


def distribution_traces(lp, data):
    """Inputs are prefix model outputs and observed-answer masks only."""
    n = lp.shape[1] - 1
    chosen = lp[:, :n].argmax(1)
    valid = data["candidate_mask"][:, :n].any(1).numpy()
    confidence = np.exp(lp[np.arange(len(lp)), chosen])
    eligible = valid & (lp[:, n] < lp[:, :n].max(1))
    result = []
    for start in range(0, len(lp), n):
        stop = start + n
        result.append({"chosen": np.where(valid[start:stop], chosen[start:stop], -1),
                       "confidence": confidence[start:stop], "can_stop": eligible[start:stop],
                       "other": np.exp(lp[start:stop, n])})
    return result


def stopping_index(history, policy, continuation=None):
    """A causal stopping decision: no gold or future-prefix aggregate input."""
    minimum, patience, cap = (policy[k] for k in ["minimum", "patience", "cap"])
    n = len(history["chosen"])
    if not 1 <= minimum <= cap <= n or patience < 1:
        raise ValueError("Invalid minimum/patience/cap")
    gamma = policy["gamma"]
    chosen = np.asarray(history["chosen"])
    qualifying = np.asarray(history["can_stop"], bool) & (
        np.asarray(history["confidence"]) >= gamma if gamma is not None else False)
    for index in range(minimum - 1, cap):
        lo = index - patience + 1
        persistent = lo >= 0 and qualifying[lo:index + 1].all() and (
            chosen[lo:index + 1] == chosen[index]).all()
        if persistent:
            return index, "confidence"
        if continuation is not None and chosen[index] >= 0:
            value, tokens = continuation[index]
            if value <= policy["continuation_lambda"] * max(float(tokens), 1.):
                return index, "continuation_value"
    return cap - 1, "cap"


def select_candidate(candidates, sc, previous=None):
    """Calibration-only empirical gate. No unvalidated accuracy guarantee."""
    for row in candidates:
        row["feasible"] = row["accuracy"] >= sc - .01 - 1e-12
    eligible = [r for r in candidates if r["feasible"]]
    if eligible:
        best = min(eligible, key=lambda r: (r["mean_tokens"], -r["accuracy"], r["mean_samples"]))
    else:
        best = min(candidates, key=lambda r: (-r["accuracy"], r["mean_tokens"], r["mean_samples"]))
    # Ties retain the previous stage. No silent degradation for an infeasible stage.
    if previous is not None:
        prev = dict(previous)
        prev["feasible"] = prev["accuracy"] >= sc - .01 - 1e-12
        if prev["feasible"] and (not best["feasible"] or best["mean_tokens"] >= prev["mean_tokens"] - 1e-9):
            return prev, False
        if not prev["feasible"] and not best["feasible"]:
            return prev, False
    return best, previous is not None


def distribution_history(lp,valid=None):
    lp=np.asarray(lp);p=np.exp(lp);chosen=lp[:,:,:8].argmax(-1)
    valid=(lp[:,:,:8].max(-1)>-1e6) if valid is None else np.asarray(valid,dtype=bool)
    chosen=np.where(valid,chosen,-1)
    out=[]
    for qi in range(len(lp)):
        out.append({'chosen':chosen[qi],'confidence':p[qi,np.arange(8),chosen[qi].clip(0)],
                    'other':p[qi,:,-1], 'can_stop':valid[qi]&(lp[qi,:,-1]<lp[qi,:,:8].max(-1))})
    return out


def choose_state_candidate(candidates,reference):
    for c in candidates:c['feasible']=c['accuracy']>=reference-.01-1e-12
    feasible=[c for c in candidates if c['feasible']]
    if feasible:return min(feasible,key=lambda c:(c['mean_tokens'],-c['accuracy'],c['mean_samples']))
    return min(candidates,key=lambda c:(-c['accuracy'],c['mean_tokens'],c['mean_samples']))


def stop_indices(observed,state_lp,policy,prediction=None,confidence=None):
    """Inference accepts current-prefix arrays only, never correctness/gold."""
    n=len(state_lp);stopped=np.full(n,7,dtype=int);reason=np.full(n,'cap',dtype=object)
    alive=np.ones(n,bool);current=np.exp(state_lp[:,:,2])
    confidence=current if confidence is None else confidence
    family=policy['family'];gamma=policy.get('gamma');lam=policy.get('lambda',0)
    for t in range(1,7):
        eligible=alive&(observed['chosen'][:,t]>=0)
        high=eligible&(confidence[:,t]>=gamma) if gamma is not None else np.zeros(n,bool)
        low=np.zeros(n,bool)
        if family=='fixed4':low=eligible&(prediction[:,t,0]<=lam*prediction[:,t,1])
        if family=='bellman':low=eligible&(prediction[:,t,0]-current[:,t]<=lam*prediction[:,t,1])
        stop=high|low
        stopped[stop]=t;reason[stop]=np.where(high[stop],'confidence','continuation_value');alive[stop]=False
    return stopped,reason


def policy_grid(lambdas,gammas):
    result=[{'family':'threshold','gamma':g} for g in gammas]
    result += [{'family':'fixed4','gamma':g,'lambda':l} for l in lambdas for g in gammas]
    result += [{'family':'bellman','gamma':None,'lambda':l} for l in lambdas]
    return result
