"""Canonical state-bank adapter, observed features, and supervised targets."""
from __future__ import annotations

import copy

import numpy as np
import torch

from pgm_asc.core.config import BOOTSTRAP, DATA, SELECTOR, STATE as ROOT
from pgm_asc.core.integrity import load_tensor
from pgm_asc.core.io import read_json, read_jsonl, write_json, write_jsonl
from pgm_asc.data.banks import build_dataset, scale
from pgm_asc.models.selector import PGM, predict


REP = BOOTSTRAP / 'cache'


def prepare(split):
    out = ROOT / 'cache' / split
    if (out / 'data.pt').exists():
        return
    if split in ('train', 'dev'):
        raise FileNotFoundError(f'Canonical packed training input missing: {out / "data.pt"}. Historical raw banks were intentionally removed; restore this input or build a new training cohort.')
    if split not in ('calibration_a', 'calibration_b', 'test'):
        raise ValueError(split)
    lock = 'selection_lock.json' if split == 'test' else 'model_lock.json'
    if not (ROOT / lock).exists():
        raise ValueError('Fresh labels require ' + lock)
    source = DATA / 'state'
    gold = {r['question_id']: r['gold_answer'] for r in read_jsonl(source / f'{split}_gold.jsonl')}
    questions = read_jsonl(source / f'{split}.jsonl')
    banks = []
    for original in questions:
        q = dict(original, gold_answer=gold[original['question_id']])
        records, vectors = [], []
        for i in range(8):
            name = f"{q['question_id']}__s{i:03d}"
            r = read_json(ROOT / split / 'scored_responses' / f'{name}.json')
            if r['question_id'] != q['question_id'] or r['sample_index'] != i:
                raise ValueError('Response identity mismatch')
            records.append(r)
            with np.load(ROOT / split / 'embeddings' / f'{name}.npz', allow_pickle=False) as z:
                vectors.append(z['final_full'].copy())
        banks.append({'question': q, 'records': records, 'vectors': np.stack(vectors)})
    data, meta = build_dataset(banks, read_json(REP / 'representation.json'), 32)
    data['raw_w'] = data['w'].clone()
    ix = torch.arange(len(meta)) % 8
    data['latest_x'] = data['x'][torch.arange(len(meta)), ix].clone()
    data = scale(data, read_json(REP / 'scalers.json'))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(data, out / 'data.pt')
    write_jsonl(out / 'metadata.jsonl', meta)
    write_jsonl(out / 'questions.jsonl', questions)
    write_json(out / 'manifest.json', {'questions': len(questions), 'prefixes': 8, 'selector_capacity': 32})


def load(split):
    prepare(split)
    path = ROOT / 'cache' / split
    return load_tensor(path / 'data.pt', mmap=True), read_jsonl(path / 'metadata.jsonl')


def selector_outputs(data, split):
    out = ROOT / 'cache' / split / 'selectors.npz'
    if out.exists():
        with np.load(out, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}
    saved = load_tensor(SELECTOR / 'refits/mixture/checkpoint.pt')
    base = PGM(32, **saved['spec'])
    base.load_state_dict(saved['state_dict']); base.eval()
    result = {}
    for name, slot in [('M', None), ('S', 1), ('A', 0), ('I', 5), ('N', 1)]:
        model = copy.deepcopy(base)
        if slot is not None:
            model.mode = 'single'; model.partition = slot
        if name == 'N':
            native = load_tensor(SELECTOR / 'refits/single/checkpoint.pt')
            model.load_state_dict(native['state_dict'])
        lp, _ = predict(model, data)
        result[name] = lp[:, [*range(8), 32]].reshape(-1, 8, 9)
    np.savez_compressed(out, **result)
    return result


def observed_features(data,lp):
    """No labels accepted/read. All arrays contain only current-prefix observations."""
    n=len(lp)
    k=data['candidate_mask'][:,:8].sum(1).numpy().reshape(n,8)
    p=np.exp(lp)
    chosen=lp[:,:,:8].argmax(-1);chosen=np.where(k>0,chosen,-1)
    ordered=np.sort(p[:,:,:8],axis=-1)
    conditional=p[:,:,:8]/np.maximum(p[:,:,:8].sum(-1,keepdims=True),1e-30)
    entropy=-(conditional*np.log(np.maximum(conditional,1e-30))).sum(-1)/np.maximum(np.log(np.maximum(k,1)),1e-12)
    entropy[k<2]=0
    new=k>np.concatenate([np.zeros((n,1)),k[:,:-1]],1)
    changed=chosen!=np.concatenate([np.full((n,1),-1),chosen[:,:-1]],1)
    run=np.ones((n,8))
    for t in range(1,8):run[:,t]=np.where(changed[:,t],1,run[:,t-1]+1)
    raww=data['raw_w'].numpy().reshape(n,8,25)
    latest=data['latest_x'].numpy().reshape(n,8,9)
    extra=np.stack([ordered[:,:,-1],ordered[:,:,-2],p[:,:,-1],ordered[:,:,-1]-ordered[:,:,-2],entropy,
                    new,changed,run/np.arange(1,9)[None],latest[:,:,3]],-1)
    x=np.concatenate([raww,extra,latest],-1).astype(np.float32)
    assert x.shape==(n,8,43) and np.isfinite(x).all()
    return {'x':x,'chosen':chosen,'new':new,'changed':changed,'count':k}


def targets(data,observed):
    n=len(observed['x']);gold=data['gold_groups'].numpy().reshape(n,8,33)
    chosen=observed['chosen']
    correct=np.take_along_axis(gold,chosen.clip(0)[:,:,None],-1).squeeze(-1)&(chosen>=0)
    seen=data['target'].numpy().reshape(n,8)!=32
    states=np.where(~seen,0,np.where(correct,2,1))
    assert not (correct&~seen).any()
    return states.astype(np.int64),correct


def costs(meta):
    return np.array([r['token_cost'] for r in meta],dtype=float).reshape(-1,8)
