"""Strict reference-string grouping with sixteen prefixes and capacity32."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from pgm_asc.core.io import read_json,read_jsonl,sample_seed,log
from pgm_asc.data.banks import response_x,INPUT_KEYS
from pgm_asc.data.features import mean_std
from pgm_asc.models.partitions import OnlinePartitions,unit
from pgm_asc.policies.reasc_reference import exact_groups


def load(cfg,split,limit=None):
    root=Path(cfg['run_dir'])/split;qs=read_jsonl(Path(cfg['data_dir'])/f'{split}.jsonl')
    if limit is not None:qs=qs[:limit]
    banks=[]
    for q in qs:
        qid=q['question_id'];records=[];vectors=[]
        for i in range(16):
            rid=f'{qid}__s{i:03d}';r=read_json(root/'scored_responses'/f'{rid}.json')
            if r['sample_index']!=i or r['question_id']!=qid:raise ValueError('Response identity mismatch')
            with np.load(root/'embeddings'/f'{rid}.npz',allow_pickle=False) as e:vectors.append(e['final_full'].copy())
            records.append(r)
        labels=read_json(root/'judgments'/f'{qid}.json')
        reps,_=exact_groups([r['prediction'] for r in records])
        if reps!=labels['representatives']:raise ValueError('Judged candidate identity mismatch')
        banks.append({'question':q,'records':records,'vectors':np.stack(vectors),'labels':labels})
    return banks


def pack(records,vectors,representation,seed=42,capacity=32):
    if len(records)>capacity:raise ValueError('Capacity exceeded')
    transformed=unit(unit(vectors)-np.asarray(representation['mean'],dtype=np.float32))
    transformed[np.linalg.norm(vectors,axis=1)<=1e-8]=0
    partitions=OnlinePartitions(representation['thresholds'],seed)
    reps,groups=exact_groups([r['prediction'] for r in records]);x=response_x(records);rows=[]
    for t in range(1,len(records)+1):
        normal=partitions.update(transformed[t-1]);random=partitions.snapshot(random=True)
        ids=groups[:t];k=int(ids.max())+1;counts=np.bincount(ids,minlength=k);leader=int(counts.argmax());ordered=sorted(counts,reverse=True)+[0,0]
        p=counts/t;w=[t,1.,k,ordered[0]/t,ordered[1]/t,(ordered[0]-ordered[1])/t,float(-(p*np.log(p)).sum()),0.]
        for mask in [np.ones(t,dtype=bool),ids==leader]:
            for col in range(4):w.extend(mean_std(x[:t,col][mask] if col<3 else np.log1p([r['generated_token_count'] for r in records[:t]])[mask]))
        w.append(float(np.mean([r['truncated'] for r in records[:t]])))
        row={'x':np.zeros((capacity,9),np.float32),'ids':np.full(capacity,-1,np.int64),'response_mask':np.arange(capacity)<t,
             'candidate_mask':np.r_[np.arange(capacity)<k,True],'candidate':np.zeros((capacity+1,6),np.float32),'w':np.asarray(w,np.float32),
             't':t,'leader':leader,'representatives':reps[:k],'token_cost':sum(r['reference_chat_tokens'] for r in records[:t])}
        row['x'][:t]=x[:t];row['ids'][:t]=ids
        for prefix,snapshot in [('',normal),('random_',random)]:
            row[prefix+'sizes']=np.ones((6,capacity),np.float32);row[prefix+'sizes'][:,:t]=snapshot['sizes']
            row[prefix+'gate']=snapshot['gate'];row[prefix+'partition_mask']=snapshot['partition_mask']
        for c in range(k):
            support=x[:t][ids==c];row['candidate'][c]=[counts[c],counts[c]/t,*support[:,:3].mean(0),support[:,3].mean()]
        rows.append(row)
    return rows


def build(banks,representation,capacity=32):
    arrays={k:[] for k in INPUT_KEYS};meta=[];targets=[];gold=[]
    for qi,b in enumerate(banks):
        rows=pack(b['records'],b['vectors'],representation,sample_seed(42,b['question']['question_id'],1),capacity)
        # No labels reach pack(); gold enters only the loss/evaluation sidecars.
        labels=b['labels']['canonical_correct']
        for row in rows:
            matches=[k for k in range(len(row['representatives'])) if labels[k]]
            target=matches[0] if matches else capacity
            targets.append(target);gold.append([k in matches for k in range(capacity)]+[not matches])
            meta.append({k:row[k] for k in ['t','leader','token_cost','representatives']}|{'question_index':qi,'question_id':b['question']['question_id']})
            for key in INPUT_KEYS:arrays[key].append(row[key])
    tensors={k:torch.from_numpy(np.stack(v)) for k,v in arrays.items()}
    tensors.update(target=torch.tensor(targets,dtype=torch.long),gold_groups=torch.tensor(gold,dtype=torch.bool),
                   question_index=torch.tensor([r['question_index'] for r in meta]),t=torch.tensor([r['t'] for r in meta]))
    return tensors,meta


def traces(logp,data):
    if len(logp)%16:raise ValueError('Sixteen prefixes required')
    c=logp.shape[1]-1;chosen=logp[:,:c].argmax(1);valid=data['candidate_mask'][:,:c].any(1).numpy()
    result=[]
    for start in range(0,len(logp),16):
        s=slice(start,start+16)
        result.append({'chosen':np.where(valid[s],chosen[s],-1),'confidence':np.exp(logp[np.arange(start,start+16),chosen[s]]),
                       'can_stop':valid[s]&(logp[s,c]<logp[s,:c].max(1)),'other':np.exp(logp[s,c])})
    return result
