"""Question-level accuracy, probability metrics, and paired bootstraps."""
from __future__ import annotations

import numpy as np


def measurements(logp, data, n):
    target = data["target"].cpu().numpy()
    probabilities = np.exp(logp)
    chosen = logp[:, :n].argmax(1)
    has_answer = data["candidate_mask"][:, :n].any(1).cpu().numpy()
    correct = data["gold_groups"].cpu().numpy()[np.arange(len(chosen)), chosen] & has_answer
    nll = -logp[np.arange(len(target)), target]
    onehot = np.eye(n + 1)[target]
    brier = ((probabilities - onehot) ** 2).sum(1)
    metrics = {}
    times = data["t"].cpu().numpy()
    for t in [8, 16, 32]:
        mask = times == t
        if mask.any():
            metrics[str(t)] = {"accuracy": float(correct[mask].mean()),
                               "nll": float(nll[mask].mean()), "brier": float(brier[mask].mean()),
                               "unobserved_gold_fraction": float((target[mask] == n).mean())}
    primary = np.isin(times, [8, 16, 32])
    return {"prefixes": metrics, "selection_nll": float(nll[primary].mean()),
            "all_prefix_nll": float(nll.mean())}, nll, brier, correct


def state_metrics(lp,y):
    nll=-np.take_along_axis(lp,y[:,:,None],-1).squeeze(-1)
    prob=np.exp(lp)
    onehot=np.eye(3)[y]
    brier=((prob-onehot)**2).sum(-1)
    result={'selection_nll':float(nll[:,[1,3,7]].mean()),'prefixes':{}}
    for t in [2,4,8]:
        i=t-1
        result['prefixes'][str(t)]={'nll':float(nll[:,i].mean()),'brier':float(brier[:,i].mean()),
            'state_accuracy':float((lp[:,i].argmax(-1)==y[:,i]).mean()),
            'coverage_Brier':float(np.square((1-prob[:,i,0])-(y[:,i]!=0)).mean()),
            'correctness_Brier':float(((prob[:,i,2]-(y[:,i]==2))**2).mean()),
            'state_counts':np.bincount(y[:,i],minlength=3).tolist()}
    return result


def paired(a,b,coverage=.95,repetitions=10000):
    delta=np.asarray(a,dtype=float)-np.asarray(b,dtype=float);rng=np.random.default_rng(20261105)
    # Batches avoid a large bootstrap index allocation for many endpoints.
    means=np.concatenate([delta[rng.integers(len(delta),size=(min(500,repetitions-i),len(delta)))].mean(1)
                          for i in range(0,repetitions,500)])
    lo=(1-coverage)/2
    return {'difference':float(delta.mean()),'interval':np.quantile(means,[lo,1-lo]).tolist(),'coverage':coverage,'questions':len(delta)}


def policy_contrast(a,b):
    return {field:paired([r[field] for r in a['rows']],[r[field] for r in b['rows']]) for field in ['correct','token_cost']}


def bootstrap_difference(a, b):
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(42)
    resampled = delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(1)
    return {"difference": float(delta.mean()), "ci95": np.quantile(resampled, [.025, .975]).tolist(),
            "questions": len(delta), "repetitions": 5000}
