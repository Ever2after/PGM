"""Persistent resumable GPU4-7 supervisor; test is gated by policy lock."""
from __future__ import annotations
import argparse,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from filelock import FileLock
from pgm_asc.core.io import read_json,read_jsonl,write_json,utc_now,log
from pgm_asc.core.integrity import file_sha,verify_lock


def update(cfg,state,phase,**extra):
    root=Path(cfg['run_dir']);progress={}
    for s in ['train','dev','calibration','math500','fresh']:
        base=root/s
        progress[s]={'generated_questions':len(list((base/'questions').glob('*.json'))),
                     'scored_responses':len(list((base/'scored_responses').glob('*.json'))),
                     'judged_questions':len(list((base/'judgments').glob('*.json')))}
    status={'state':state,'phase':phase,'supervisor_pid':os.getpid(),'gpus':cfg['gpus'],'updated_at':utc_now(),'progress':progress,**extra}
    write_json(root/'status.json',status)
    lines=['# ReASC/PGM run status','',f'Updated: {status["updated_at"]}',f'State: {state}; phase: {phase}; supervisor PID: {os.getpid()}','',
           '| Split | Generated questions | Scored responses | Judged questions |','|---|---:|---:|---:|']
    for s,p in progress.items():lines.append(f'| {s} | {p["generated_questions"]} | {p["scored_responses"]} | {p["judged_questions"]} |')
    lines+=['','GPU4,5: independent TP1 vLLM replicas, batch128 each; GPU6: TF features; GPU7: xVerify.','',
            'Order: 16-dev smoke → calibration/train/dev → train/select/lock → MATH500/fresh evaluation.',
            'Checkpoint and policy selection never use test outcomes. Existing Frozen v2 artifacts are retained.',
            'This is a controlled reproduction; pinned xVerify-Ia and execution adaptations are documented in configs/reasc_comparison.json.']
    (root/'progress.md').write_text('\n'.join(lines)+'\n')
    return status


def smoke_ready(cfg):
    root=Path(cfg['run_dir'])/'dev';qs=read_jsonl(Path(cfg['data_dir'])/'dev.jsonl')[:16]
    return all((root/'questions'/f"{q['question_id']}.json").exists() and (root/'judgments'/f"{q['question_id']}.json").exists()
               and all((root/'scored_responses'/f"{q['question_id']}__s{i:03d}.json").exists() for i in range(16)) for q in qs)


def smoke_audit(cfg):
    import torch
    torch.set_num_threads(4)
    from pgm_asc.data.reasc_bank import load,build,traces
    from pgm_asc.data.banks import fit_representation
    from pgm_asc.models.selector import PGM,predict
    root=Path(cfg['run_dir']);banks=load(cfg,'dev',16);records=[r for b in banks for r in b['records']]
    off_top=0;max_native_difference=0.
    for r in records:
        with np.load(root/'dev/top_logprobs'/f"{r['response_id']}.npz",allow_pickle=False) as z:
            token_matrix=z['token_ids'];logprob_matrix=z['logprobs']
            for i,token in enumerate(r['token_ids']):
                where=np.flatnonzero(token_matrix[i]==token)
                if len(where)!=1:raise ValueError('Sampled token ID alignment error')
                value=float(logprob_matrix[i,where[0]]);max_native_difference=max(max_native_difference,abs(value-r['native_sample_logprobs'][i]))
                off_top+=int(token!=int(token_matrix[i,np.argmax(logprob_matrix[i])]))
        if len(r['token_ids'])!=r['generated_token_count'] or r['generated_token_count']>1024:raise ValueError('Token length mismatch')
        if not all(np.isfinite(r[k]) for k in ['reasoning_mean_logprob','answer_mean_logprob','reasoning_mean_entropy']):raise ValueError('Nonfinite TF feature')
    if off_top==0:raise ValueError('No off-argmax sampled tokens exercised')
    if max_native_difference>1e-5:raise ValueError('Stored sampled logprobs mismatch')
    # Diagnostic-only dev fit validates shapes; never persisted or reused for primary preprocessing.
    data,_=build(banks,fit_representation(banks));lp,_=predict(PGM(32),data);h=traces(lp,data)
    assert len(h)==16 and all(len(x['chosen'])==16 for x in h)
    elapsed=sum(r['native_generation_seconds_share']+r['extraction_inline']['seconds_share']+r['extraction_final']['seconds_share'] for r in records)
    bytes_used=sum(p.stat().st_size for p in (root/'dev').rglob('*') if p.is_file())
    result={'passed':True,'created_at':utc_now(),'questions':16,'responses':256,'sampled_off_argmax_tokens':off_top,
            'stored_native_lp_max_difference':max_native_difference,'mean_question_generation_seconds':elapsed/16,
            'rough_generation_hours_2128questions':elapsed/16*2128/3600,'dev_bytes_at_audit':bytes_used,
            'rough_bank_GiB':bytes_used/16*2128/2**30,'tf_lp_difference_note':'Top-dictionary native and raw full-vocabulary TF quantities are recorded separately, not asserted identical.',
            'body_box_missing_responses':sum(not r['answer_token_count'] for r in records),'truncated_responses':sum(r['truncated'] for r in records),
            'dev_smoke_fit_not_used_for_primary':True}
    write_json(root/'smoke_validation.json',result);return result


def run_group(cfg,split,limit=None):
    root=Path(cfg['run_dir']);python=str(Path('.venv-reasc/bin/python').absolute());jobs=[]
    commands=[('generation',cfg['generation_gpus'],['-m','pgm_asc.collection.reasc_bank','collect']),
              ('features',[cfg['feature_gpu']],['-m','pgm_asc.collection.reasc_bank','features','--follow']),
              ('judge',[cfg['judge_gpu']],['-m','pgm_asc.collection.reasc_judge','--follow'])]
    if cfg.get('generation_runtime')=='tp1_batched_replicas':
        commands=commands[1:]
        for shard,gpu in enumerate(cfg['generation_gpus']):
            commands.insert(shard,(f'generation_gpu{gpu}',[gpu],['-m','pgm_asc.collection.reasc_fast','--shard',str(shard),'--shards',str(len(cfg['generation_gpus']))]))
    try:
        for name,gpus,args in commands:
            target=root/'logs'/f'{split}_{name}.log'
            target.parent.mkdir(exist_ok=True);handle=target.open('a')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            args+=['--config','configs/reasc_comparison.json','--split',split]
            if limit is not None:args+=['--limit',str(limit)]
            process=subprocess.Popen([python,*args],env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((name,process,handle))
        while True:
            for name,p,_ in jobs:
                if p.poll() not in [None,0]:raise RuntimeError(f'{split}/{name} exited {p.returncode}; see logs/{split}_{name}.log')
            update(cfg,'running',f'bank_{split}',workers={name:p.pid for name,p,_ in jobs})
            if all(p.poll()==0 for _,p,_ in jobs):break
            time.sleep(15)
    finally:
        for _,p,handle in jobs:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
            handle.close()


def main(cfg,adopt_smoke=False):
    root=Path(cfg['run_dir']);root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'.supervisor.lock'),timeout=0):
        try:
            if not (root/'smoke_validation.json').exists():
                if adopt_smoke:
                    while not smoke_ready(cfg):
                        update(cfg,'running','smoke_adopted')
                        # Fail rather than hang if an adopted stage leaves a traceback.
                        for stage in ['generation','features','judge']:
                            path=root/f'smoke_{stage}.log'
                            if path.exists() and 'Traceback (most recent call last)' in path.read_text()[-10000:]:
                                raise RuntimeError(f'Adopted smoke {stage} failed; see {path}')
                        time.sleep(15)
                    time.sleep(10)
                else:run_group(cfg,'dev',16)
                update(cfg,'running','smoke_audit')
                smoke_audit(cfg)
            for split in ['calibration','train','dev']:run_group(cfg,split)
            update(cfg,'running','fit_calibrate_select_lock')
            # Small PGM heads train on CPU while the large-model stages have exited.
            from pgm_asc.experiments.reasc_comparison import select
            select(cfg);verify_lock(read_json(root/'selection_lock.json'))
            for split in ['math500','fresh']:
                run_group(cfg,split);update(cfg,'running',f'evaluate_{split}')
                from pgm_asc.evaluation.reasc_comparison import evaluate
                evaluate(cfg,split)
            update(cfg,'complete','comparison_complete')
            lines=['# ReASC / PGM comparison','', 'Both datasets evaluated with policies locked before test generation.','',
                   '- [MATH500 results](math500/report.md)','- [Fresh holdout results](fresh/report.md)',
                   '- [Execution settings](../../../configs/reasc_comparison.json)','',
                   'Canonical candidate and reference-code accuracy are reported separately. Reference token savings are not an end-to-end speedup claim.']
            (root/'result_summary.md').write_text('\n'.join(lines)+'\n')
        except BaseException as e:
            write_json(root/'failure.json',{'at':utc_now(),'error':repr(e),'traceback':traceback.format_exc()});update(cfg,'failed','see_failure_json',error=repr(e));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/reasc_comparison.json');p.add_argument('--adopt-smoke',action='store_true');a=p.parse_args();main(read_json(a.config),a.adopt_smoke)
