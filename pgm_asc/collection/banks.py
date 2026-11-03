"""Cohort selection, collection subprocesses, bank audit, and sharding."""
from __future__ import annotations

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pgm_asc.core.answers import boxed_span
from pgm_asc.core.config import DATA
from pgm_asc.core.integrity import file_sha, problem_key
from pgm_asc.core.io import digest, read_json, read_jsonl, utc_now, write_json, write_jsonl
from pgm_asc.core.runtime import runtime_env


def prepare(config):
    root=Path(config['run_dir']);data=Path(config['data_dir'])
    if config['samples_per_question'] == 8 and not (root/'model_lock.json').exists():raise ValueError('Model lock required')
    path=data/'manifest.json'
    if path.exists():
        manifest=read_json(path)
        if manifest['config_hash']!=digest(config):raise ValueError('Cohort config changed')
        for p,h in manifest['input_sha256'].items():
            if file_sha(p)!=h:raise ValueError('Cohort changed')
        return manifest
    import pandas as pd
    from huggingface_hub import hf_hub_download
    excluded=set(read_json(DATA/'excluded_problem_hashes.json')['hashes']);source_hash={}
    for p in Path('artifacts/data').rglob('*.jsonl'):
        if data in p.parents:continue
        keys={problem_key(q['problem']) for q in read_jsonl(p) if 'problem' in q}
        if keys:excluded.update(keys);source_hash[str(p)]=file_sha(p)
    eligible=[];seen=set()
    for subject in config['subjects']:
        p=hf_hub_download(config['dataset_id'],f'{subject}/test-00000-of-00001.parquet',repo_type='dataset',
                           revision=config['dataset_revision'],cache_dir=config['cache_dir'],local_files_only=True)
        for i,row in enumerate(pd.read_parquet(p).to_dict('records')):
            key=problem_key(row['problem'])
            if key in excluded or key in seen:continue
            seen.add(key);level=re.fullmatch(r'Level ([1-5])',str(row['level']).strip())
            eligible.append({'question_id':f'pgmasc_test_{subject}_{i:04d}_{key[:8]}','problem':row['problem'],
                             'subject':subject,'source_split':'test','source_index':i,
                             'level':int(level.group(1)) if level else None,'contains_asy':'[asy]' in row['problem'].lower(),
                             'solution':row['solution']})
    required=sum(config['followup_sizes'].values())
    if len(eligible)<required:raise ValueError('Insufficient untouched questions')
    order=np.random.default_rng(config['selection_seed']).permutation(len(eligible));start=0;hashes={};splits={}
    for split in ['calibration_a','calibration_b','test']:
        size=config['followup_sizes'][split];qs=[];gold=[]
        for i in order[start:start+size]:
            q=dict(eligible[int(i)]);box=boxed_span(q.pop('solution'))
            if not box:raise ValueError('Selected gold missing')
            qs.append(q);gold.append({'question_id':q['question_id'],'gold_answer':box['answer']})
        start+=size
        write_jsonl(data/f'{split}.jsonl',qs);write_jsonl(data/f'{split}_gold.jsonl',gold)
        hashes.update({str(data/f'{n}.jsonl'):file_sha(data/f'{n}.jsonl') for n in [split,split+'_gold']})
        splits[split]={'questions':size,'asy':sum(q['contains_asy'] for q in qs)}
    from transformers import AutoTokenizer
    snapshot=Path(config['cache_dir'])/'models--google--gemma-3-4b-it/snapshots'/config['model_revision']
    tokenizer=AutoTokenizer.from_pretrained(snapshot)
    maximum=max(len(tokenizer.apply_chat_template([{'role':'user','content':q['problem']+'\n\n'+config['prompt_suffix']}],
                   tokenize=True,add_generation_prompt=True)) for split in splits for q in read_jsonl(data/f'{split}.jsonl'))
    if maximum+config['max_new_tokens']>config['max_model_len']:raise ValueError('Context limit')
    manifest={'created_at':utc_now(),'config_hash':digest(config),'eligible':len(eligible),'splits':splits,
              'input_sha256':hashes,'exclusion_source_sha256':source_hash,'normalized_overlap':0,'max_prompt_tokens':maximum,
              'model_lock_existed_before_cohort':True,'holdout_evaluated':False}
    write_json(path,manifest);write_json(root/'cohort_lock.json',manifest)
    return manifest


def audit_bank(config, split):
    root = Path(config['run_dir']) / split
    n = config['samples_per_question']
    questions = read_jsonl(Path(config['data_dir']) / f'{split}.jsonl')
    expected = {(q['question_id'], i) for q in questions for i in range(n)}
    seen = set(); errors = []; failures = truncated = 0
    for p in (root / 'scored_responses').glob('*.json'):
        r = read_json(p); identity = (r['question_id'], r['sample_index'])
        if identity in seen:
            raise ValueError('Duplicate response identity')
        seen.add(identity)
        errors.append(r['raw_logprob_backend_mean_abs_difference'])
        failures += r['extraction_failed']; truncated += r['truncated']
    if not expected or seen != expected:
        raise ValueError(f'Incomplete/extra bank membership: {split}')
    if not np.isfinite(errors).all() or max(errors) >= .05:
        raise ValueError('Raw confidence backend validation failed')
    result = {'created_at': utc_now(), 'responses': len(seen), 'questions': len(questions),
              'extraction_failures': failures, 'truncated': truncated,
              'mean_raw_lp_difference': float(np.mean(errors)), 'max_raw_lp_difference': max(errors), 'passed': True}
    write_json(root / 'bank_audit.json', result)
    return result


def collect_one(config, split, gpu):
    root = Path(config['run_dir']); root.mkdir(parents=True, exist_ok=True)
    if (root / split / 'bank_audit.json').exists():
        audit_bank(config, split)
        return
    config_path = root / f'runtime_config_{split}.json'
    write_json(config_path, config)
    for module, vllm, option in [('vllm', True, '--split'),
                                  ('teacher_forcing', False, '--split'),
                                  ('confidence', False, '--splits')]:
        python = Path.cwd() / ('.venv-vllm/bin/python' if vllm else '.venv/bin/python')
        args = [str(python), '-u', '-m', f'pgm_asc.collection.{module}', '--config', str(config_path), option, split]
        with (root / f'{split}_{module}.log').open('a') as output:
            subprocess.run(args, cwd=Path.cwd(), env=runtime_env(Path.cwd(), gpu, vllm),
                           stdout=output, stderr=subprocess.STDOUT, check=True)
    audit_bank(config, split)


def calibration(config):
    prepare(config)
    gpus = config.get('bank_gpus', [4, 5])
    with ThreadPoolExecutor(len(gpus)) as pool:
        tasks = [pool.submit(collect_one, config, split, gpus[i % len(gpus)])
                 for i, split in enumerate(['calibration_a', 'calibration_b'])]
        for future in tasks:
            future.result()


def holdout(config):
    root = Path(config['run_dir']); data = Path(config['data_dir'])
    if not (root / 'selection_lock.json').exists():
        raise ValueError('Policy lock required before holdout collection')
    if (root / 'test/bank_audit.json').exists():
        audit_bank(config, 'test'); return
    questions = read_jsonl(data / 'test.jsonl')
    gpus = config.get('bank_gpus', [4, 5]); children = []
    subsets = np.array_split(np.arange(len(questions)), len(gpus))
    for gpu, indices in zip(gpus, subsets):
        child = dict(config, data_dir=str(data / 'shards' / f'gpu{gpu}'),
                     run_dir=str(root / 'shards' / f'gpu{gpu}'))
        write_jsonl(Path(child['data_dir']) / 'test.jsonl', [questions[int(i)] for i in indices])
        children.append((child, gpu))
    with ThreadPoolExecutor(len(gpus)) as pool:
        tasks = [pool.submit(collect_one, child, 'test', gpu) for child, gpu in children]
        for future in tasks:
            future.result()
    for folder in ['responses', 'scored_responses', 'embeddings', 'baseline_features']:
        destination = root / 'test' / folder; destination.mkdir(parents=True, exist_ok=True)
        for child, _ in children:
            for p in (Path(child['run_dir']) / 'test' / folder).glob('*__s*'):
                target = destination / p.name
                if target.is_symlink() or target.exists():
                    raise ValueError(f'Duplicate merged response: {target}')
                target.symlink_to(os.path.relpath(p.resolve(), target.parent.resolve()))
    audit_bank(config, 'test')
