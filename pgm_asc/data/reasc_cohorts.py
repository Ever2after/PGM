"""Locked, original-split-verified cohorts for the ReASC-setting comparison."""
from __future__ import annotations
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
from pgm_asc.core.io import read_json, read_jsonl, write_json, write_jsonl, digest, utc_now
from pgm_asc.core.integrity import problem_key, file_sha

DATA = Path('artifacts/data/reasc_pgm')
RUN = Path('artifacts/runs/reasc_pgm')


def prepare():
    manifest = DATA / 'manifest.json'
    if manifest.exists():
        result = read_json(manifest)
        for p, sha in result['files'].items():
            if file_sha(p) != sha: raise ValueError('Cohort changed: '+p)
        return result
    source = Path('third_party/reasc_official'); pool = read_jsonl(source/'math500/train.jsonl')
    test = read_jsonl(source/'math500/test.jsonl')
    historical = set(read_json('artifacts/data/pgm_asc/excluded_problem_hashes.json')['hashes'])
    testkeys = {problem_key(q['problem']) for q in test}
    selected = random.Random(0).sample(pool, 128)
    calibration_keys = {problem_key(q['problem']) for q in selected}
    if len(calibration_keys) != 128 or calibration_keys & testkeys: raise ValueError('Reference calibration overlap/duplicate')
    # Cross-check actual original HF train/test question membership, not just archive IDs.
    cache = Path('artifacts/hf_cache/datasets--EleutherAI--hendrycks_math/snapshots/21a5633873b6a120296cce3e2df9d5550074f4a3')
    originals = {}
    for split in ['train', 'test']:
        keys = set()
        for p in cache.glob('*/'+split+'-*.parquet'):
            keys.update(problem_key(s) for s in pd.read_parquet(p, columns=['problem'])['problem'])
        if not keys: raise ValueError('Original MATH source unavailable')
        originals[split] = keys
    eligible = {'train': [], 'test': []}; seen = set()
    for q in pool:
        key = problem_key(q['problem']); split = q['unique_id'].split('/')[0]
        if key in seen or key in historical or key in calibration_keys or key in testkeys: continue
        if key not in originals[split]: raise ValueError('Original split mismatch')
        eligible[split].append(q); seen.add(key)
    rng = np.random.default_rng(20261102)
    order = rng.permutation(len(eligible['train']))
    cohorts = {'train': [eligible['train'][i] for i in order[:800]],
               'dev': [eligible['train'][i] for i in order[800:1000]],
               'calibration': selected, 'math500': test,
               'fresh': [eligible['test'][i] for i in rng.permutation(len(eligible['test']))[:500]]}
    expected = {'train':800,'dev':200,'calibration':128,'math500':500,'fresh':500}; used = set(); files = {}
    for split, qs in cohorts.items():
        if len(qs) != expected[split]: raise ValueError('Insufficient '+split)
        inputs = []; gold = []
        for q in qs:
            key = problem_key(q['problem'])
            if key in used: raise ValueError('Cross-cohort duplicate')
            used.add(key); qid = 'reasc_'+split+'_'+key[:16]
            inputs.append({'question_id':qid,'problem':q['problem'],'source_id':q['unique_id'],
                           'source_split':q['unique_id'].split('/')[0],'level':q.get('level'),'subject':q.get('subject'),
                           'contains_asy':'[asy]' in q['problem'].lower(),'problem_key':key})
            gold.append({'question_id':qid,'gold_answer':q['answer']})
        write_jsonl(DATA/f'{split}.jsonl',inputs);write_jsonl(DATA/f'{split}_gold.jsonl',gold)
        for suffix in ['', '_gold']:
            p=DATA/f'{split}{suffix}.jsonl';files[str(p)]=file_sha(p)
    result={'created_at':utc_now(),'selection_seed':20261102,'calibration_rng':'isolated Python Random(0); selected before model initialization',
            'rng_deviation':'Reference samples after LLM initialization; IDs are pinned here before model initialization to prevent hidden RNG consumption.',
            'questions':expected,'files':files,'eligible':{s:len(q) for s,q in eligible.items()},'normalized_overlap':0,
            'train_dev_original_split':'train','fresh_original_split':'test','fresh_historical_overlap':0,
            'benchmark_previously_observed':True,'samples_per_question':16,'planned_responses':34048,
            'source_sha256':{str(source/'math500'/name):file_sha(source/'math500'/name) for name in ['train.jsonl','test.jsonl']},
            'historical_registry_sha256':file_sha('artifacts/data/pgm_asc/excluded_problem_hashes.json')}
    write_json(manifest,result);write_json(RUN/'cohort_lock.json',result)
    return result

if __name__=='__main__': print(prepare())
