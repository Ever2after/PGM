"""Prospectively fixed datasets and run matrix for PGM necessity study."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from pgm_asc.core.io import read_json,read_jsonl,write_json,write_jsonl,utc_now
from pgm_asc.core.integrity import problem_key,file_sha

ROOT=Path('artifacts/data/pgm_followup')
RUN=Path('artifacts/runs/pgm_followup')


def emit(dataset,split,rows,original_split):
    inputs=[];gold=[]
    for i,q in enumerate(rows):
        p=q.get('problem',q.get('question'));key=problem_key(p)
        qid=q.get('question_id',f'study_{dataset}_{split}_{key[:16]}')
        inputs.append({'question_id':qid,'problem':p,'source_split':q.get('source_split',original_split),
                       'source_id':q.get('source_id',q.get('unique_id',str(i))),'problem_key':key})
        answer=q.get('gold_answer',q.get('answer'))
        if dataset=='gsm8k':answer=answer.split('####')[-1].strip()
        gold.append({'question_id':qid,'gold_answer':answer})
    path=ROOT/dataset;write_jsonl(path/f'{split}.jsonl',inputs);write_jsonl(path/f'{split}_gold.jsonl',gold)
    return inputs


def prepare():
    manifest=ROOT/'manifest.json'
    if manifest.exists():
        x=read_json(manifest)
        for p,sha in x['artifact_sha256'].items():
            if file_sha(p)!=sha:raise ValueError('Changed cohort '+p)
        return x
    prior=Path('artifacts/data/reasc_pgm');used=set();counts={}
    for split in ['train','dev','calibration','math500','fresh']:
        questions=read_jsonl(prior/f'{split}.jsonl');gold={q['question_id']:q['gold_answer'] for q in read_jsonl(prior/f'{split}_gold.jsonl')}
        used.update(problem_key(q['problem']) for q in questions)
        target='previous_fresh' if split=='fresh' else split
        rows=[q|{'gold_answer':gold[q['question_id']]} for q in questions]
        emit('math',target,rows,'mixed' if split=='calibration' else 'train' if split in ['train','dev'] else 'test')
    excluded=used|set(read_json('artifacts/data/pgm_asc/excluded_problem_hashes.json')['hashes'])
    pool=read_jsonl('third_party/reasc_official/math500/train.jsonl');eligible=[];seen=set()
    for q in pool:
        key=problem_key(q['problem'])
        if q['unique_id'].startswith('test/') and key not in excluded and key not in seen:eligible.append(q);seen.add(key)
    if len(eligible)<500:raise ValueError(f'Only {len(eligible)} unused MATH test questions remain')
    rng=np.random.default_rng(20261103);confirm=[eligible[i] for i in rng.permutation(len(eligible))[:500]]
    emit('math','confirm',confirm,'test');counts['math_confirm_eligible']=len(eligible)
    asset=read_json(RUN/'gsm_asset.json');path=Path(asset['path'])/'main'
    gsm={s:pd.concat([pd.read_parquet(p) for p in sorted(path.glob(s+'-*.parquet'))]).to_dict('records') for s in ['train','test']}
    seen=set();train=[]
    # Keep all official test rows; exclude textual duplicates of them from fitting.
    testkeys={problem_key(r['question']) for r in gsm['test']}
    for q in gsm['train']:
        key=problem_key(q['question'])
        if key not in seen and key not in testkeys:train.append(q);seen.add(key)
    order=np.random.default_rng(20261103).permutation(len(train));offset=0
    for split,n in [('calibration',128),('train',800),('dev',200)]:
        emit('gsm8k',split,[train[i] for i in order[offset:offset+n]],'train');offset+=n
    emit('gsm8k','test',gsm['test'],'test')
    fitting_keys=set()
    for dataset in ['math','gsm8k']:
        for split in ['train','dev','calibration']:fitting_keys.update(q['problem_key'] for q in read_jsonl(ROOT/dataset/f'{split}.jsonl'))
    omni=read_jsonl('third_party/reasc_official/omnimath/test.jsonl');unique=[];seen=set();duplicates=overlaps=0
    for i,q in enumerate(omni):
        key=problem_key(q['problem'])
        if key in seen:duplicates+=1;continue
        seen.add(key)
        if key in fitting_keys:overlaps+=1;continue
        unique.append(q|{'source_id':f'archive_row_{i}'})
    emit('omnimath','test',unique,'test')
    for dataset in ['math','gsm8k','omnimath']:
        splits={p.stem:read_jsonl(p) for p in (ROOT/dataset).glob('*.jsonl') if not p.stem.endswith('_gold')}
        keys={s:{q['problem_key'] for q in qs} for s,qs in splits.items()}
        for a in keys:
            for b in keys:
                if a<b and keys[a]&keys[b]:raise ValueError(f'Cross-split overlap {dataset}/{a}/{b}')
        counts[dataset]={s:len(qs) for s,qs in splits.items()}
    files=list(ROOT.rglob('*.jsonl'))
    x={'created_at':utc_now(),'selection_seed':20261103,'counts':counts,'math_confirm_excludes_all_prior_registered_questions':True,
       'math500_and_previous_fresh_are_reused_exploratory_benchmarks':True,'omni_source_rows':len(omni),'omni_duplicate_rows_removed':duplicates,
       'omni_overlap_with_fitting_removed':overlaps,'omni_is_zero_shot_transfer_of_MATH_heads':True,'gsm_source':asset,
       'artifact_sha256':{str(p):file_sha(p) for p in files},'original_reasc_lock_sha256':file_sha('artifacts/runs/reasc_pgm/selection_lock.json')}
    write_json(manifest,x);return x


def matrix():
    manifest=prepare();pins=read_json(RUN/'source_pins.json')['sources'];base=read_json('configs/reasc_comparison.json')
    models=[('gemma4b',base['model_id'],base['model_revision'],{'temperature':1.,'top_p':.95,'top_k':64,'repetition_penalty':1.}),
            ('qwen3b','Qwen/Qwen2.5-3B-Instruct',pins['Qwen/Qwen2.5-3B-Instruct']['revision'],pins['Qwen/Qwen2.5-3B-Instruct']['generation_config']),
            ('qwen7b','Qwen/Qwen2.5-7B-Instruct',pins['Qwen/Qwen2.5-7B-Instruct']['revision'],pins['Qwen/Qwen2.5-7B-Instruct']['generation_config'])]
    jobs=[]
    # Interleave datasets/models so a second model and dataset get exercised early.
    schedule=[('gemma4b','math'),('qwen3b','math'),('gemma4b','gsm8k'),('qwen3b','gsm8k'),('qwen7b','math'),('qwen7b','gsm8k'),
              ('gemma4b','omnimath'),('qwen3b','omnimath'),('qwen7b','omnimath')]
    for tag,dataset in schedule:
        _,mid,rev,sampling=next(m for m in models if m[0]==tag)
        cfg={k:base[k] for k in ['cache_dir','judge','gpus','generation_gpus','feature_gpu','judge_gpu','train']}
        cfg.update(model_id=mid,model_revision=rev,model_tag=tag,dataset=dataset,run_dir=str(RUN/f'{tag}_{dataset}'),data_dir=str(ROOT/dataset),
                   max_new_tokens=2048 if dataset=='omnimath' else 1024,evaluation_splits=['math500','confirm','previous_fresh'] if dataset=='math' and tag=='gemma4b' else ['math500','confirm'] if dataset=='math' else ['test'],
                   train_seeds=[42,43,44],capacity=32,samples_per_question=16,rounded_parameters={'gemma4b':4e9,'qwen3b':3e9,'qwen7b':7e9}[tag],
                   label='PGM necessity study: matched supervision; no continuation; pinned official-model defaults')
        cfg.update({k:sampling[k] for k in ['temperature','top_p','top_k','repetition_penalty']})
        if dataset=='omnimath':cfg['transfer_from']=str(RUN/f'{tag}_math')
        p=Path('configs/pgm_followup')/f'{tag}_{dataset}.json';write_json(p,cfg);jobs.append(str(p))
        if tag=='gemma4b' and dataset=='math':
            out=Path(cfg['run_dir']);out.mkdir(parents=True,exist_ok=True)
            for split,source in [('train','train'),('dev','dev'),('calibration','calibration'),('math500','math500'),('previous_fresh','fresh')]:
                dest=out/split
                if not dest.exists():dest.symlink_to((Path('artifacts/runs/reasc_pgm')/source).resolve(),target_is_directory=True)
    protocol={'created_at':utc_now(),'jobs':jobs,'gpus':[4,5,6,7],'train_seeds':[42,43,44],
              'primary_seed':42,'primary_operating_point':'dev-selected no-value; minimum2 patience1 cap16; minimize total output tokens at SC16 dev accuracy minus1pp',
              'secondary_operating_point':'global-T calibrated fixed0.95; same min/patience/cap',
              'primary_ablation_contrasts':['mixture vs alpha1','mixture vs random','mixture vs candidate_confidence','mixture vs deepsets_embedding'],
              'multiple_testing':'Holm across four primary structure contrasts per dataset/model; raw CIs also reported',
              'primary_new_MATH_holdout':'confirm500; existing math500/previous_fresh exploratory',
              'omni_protocol':'Transfer every model-specific MATH head, calibrator, threshold and baseline offline calibration without Omni labels',
              'cohort_manifest_sha256':file_sha(ROOT/'manifest.json')}
    write_json('configs/pgm_followup.json',protocol);return protocol


if __name__=='__main__':print(matrix())
