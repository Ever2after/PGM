"""Gold-separated replay metrics for fixed selector and state policies."""
from __future__ import annotations

import numpy as np

from pgm_asc.data.state import costs
from pgm_asc.policies.calibration import calibrate_state as calibrate
from pgm_asc.policies.stopping import stop_indices, stopping_index


def replay_policy(histories, data, metadata, policy, continuation=None):
    n = len(histories[0]["chosen"])
    gold = data["gold_groups"].numpy()
    rows = []
    for qi, history in enumerate(histories):
        index, reason = stopping_index(history, policy, None if continuation is None else continuation[qi])
        chosen = int(history["chosen"][index])
        correct = chosen >= 0 and bool(gold[qi * n + index, chosen])
        rows.append({"question_id": metadata[qi * n]["question_id"], "t": index + 1,
                     "chosen": chosen, "correct": bool(correct), "reason": reason,
                     "token_cost": metadata[qi * n + index]["token_cost"]})
    return {"accuracy": float(np.mean([r["correct"] for r in rows])),
            "mean_tokens": float(np.mean([r["token_cost"] for r in rows])),
            "mean_samples": float(np.mean([r["t"] for r in rows])), "rows": rows}


def sc_accuracy(data, metadata, n=32):
    gold = data["gold_groups"].numpy()
    final = np.arange(n - 1, len(metadata), n)
    chosen = np.asarray([r["leader"] for r in metadata])[final]
    valid = data["candidate_mask"][final, :n].any(1).numpy()
    return float((gold[final, chosen] & valid).mean())


def decision_metrics(t,reason,chosen,data,metadata):
    n=len(t);gold=data['gold_groups'].numpy().reshape(n,8,33);qq=np.arange(n)
    answer=chosen[qq,t];correct=(answer>=0)&gold[qq,t,answer.clip(0)]
    token=costs(metadata)[qq,t]
    rows=[{'question_id':metadata[i*8]['question_id'],'t':int(t[i]+1),'chosen':int(answer[i]),
           'correct':bool(correct[i]),'token_cost':float(token[i]),'reason':str(reason[i])} for i in range(n)]
    return {'accuracy':float(correct.mean()),'mean_tokens':float(token.mean()),'mean_samples':float((t+1).mean()),'rows':rows}


def run_state_policy(ctx,recipe,split,policy,temperature):
    lp=ctx.state(recipe,split);obs=ctx.observed(recipe['selector'],split)
    values=None
    if policy['family']=='fixed4':values=ctx.values(recipe,split)['fixed4']
    if policy['family']=='bellman':values=ctx.values(recipe,split)['bellman_'+str(policy['lambda'])]
    confidence=np.exp(calibrate(lp,temperature)[:,:,2])
    t,reason=stop_indices(obs,lp,policy,values,confidence)
    data,meta=ctx.data(split)
    return decision_metrics(t,reason,obs['chosen'],data,meta)


def history_metrics(histories,data,meta,gamma):
    n=len(histories);chosen=np.stack([h['chosen'] for h in histories]);t=np.full(n,7,int);reason=np.full(n,'cap',object)
    alive=np.ones(n,bool)
    for i in range(1,7):
        hit=np.array([h['can_stop'][i] and h['confidence'][i]>=gamma for h in histories]) if gamma is not None else np.zeros(n,bool)
        use=alive&hit;t[use]=i;reason[use]='confidence';alive[use]=False
    return decision_metrics(t,reason,chosen,data,meta)
