"""GPU data queue plus independent CPU fit/lock worker, resumable by cell."""
import argparse,os,signal,subprocess,time,traceback
from pathlib import Path
from filelock import FileLock
from pgm_asc.core.io import read_json,read_jsonl,write_json,utc_now,log
from pgm_asc.core.integrity import verify_lock,file_sha

ROOT=Path('artifacts/runs/pgm_followup')
PYTHON=str(Path('.venv-reasc/bin/python').absolute())


def bank_complete(cfg,split):
    root=Path(cfg['run_dir'])/split
    for q in read_jsonl(Path(cfg['data_dir'])/f'{split}.jsonl'):
        qid=q['question_id']
        if not (root/'questions'/f'{qid}.json').exists() or not (root/'judgments'/f'{qid}.json').exists():return False
        if not all((root/'scored_responses'/f'{qid}__s{i:03d}.json').exists() for i in range(16)):return False
    return True


def update(protocol,phase,**extra):
    cells={}
    for path in protocol['jobs']:
        cfg=read_json(path);root=Path(cfg['run_dir']);state={}
        for split in ['train','dev','calibration']+cfg['evaluation_splits']:
            if cfg.get('transfer_from') and split in ['train','dev','calibration']:continue
            if not (Path(cfg['data_dir'])/f'{split}.jsonl').exists():continue
            state[split]={'questions':len(read_jsonl(Path(cfg['data_dir'])/f'{split}.jsonl')),
                          'generated':len(list((root/split/'questions').glob('*.json'))),
                          'scored_responses':len(list((root/split/'scored_responses').glob('*.json'))),
                          'judged':len(list((root/split/'judgments').glob('*.json'))),
                          'evaluated':(root/split/'study_evaluation.json').exists()}
        cells[root.name]={'policy_locked':(root/'selection_lock.json').exists(),'fit':read_json(root/'fit_status.json') if (root/'fit_status.json').exists() else None,'splits':state}
    status={'updated_at':utc_now(),'phase':phase,'supervisor_pid':os.getpid(),'gpus':[4,5,6,7],'cells':cells,**extra}
    write_json(ROOT/'status.json',status)
    lines=['# PGM follow-up study','',f'Updated: {status["updated_at"]}; phase: {phase}','',
           '| Cell | Policy locked | Fits complete | Evaluation complete |','|---|---:|---:|---:|']
    for cell,s in cells.items():lines.append(f"| {cell} | {s['policy_locked']} | {(s['fit'] or {}).get('completed_fits',0)} | {sum(x['evaluated'] for x in s['splits'].values())} |")
    lines+=['','GPU4/5 TP1 vLLM replicas; GPU6 exact-token features; GPU7 xVerify; CPU matched-budget fitting.',
            'Each cell: 11 independently trained arms × seeds42/43/44 + 3 same-head interventions. All use no-value stopping.',
            'MATH confirm500 is the new holdout. Existing MATH500/previous_fresh are explicitly exploratory. Omni uses frozen MATH transfer.']
    (ROOT/'progress.md').write_text('\n'.join(lines)+'\n')


def spawn(args,gpus,logpath):
    logpath.parent.mkdir(parents=True,exist_ok=True);handle=logpath.open('a')
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)))
    p=subprocess.Popen([PYTHON,*args],stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    return p,handle


def stop(process):
    p,handle=process
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
    handle.close()


def run_bank(protocol,path,split,fit_worker):
    cfg=read_json(path)
    if bank_complete(cfg,split):return
    if (Path(cfg['run_dir'])/split).is_symlink():raise ValueError('Never write linked original bank')
    if split not in ['train','dev','calibration']:verify_lock(read_json(Path(cfg['run_dir'])/'selection_lock.json'))
    commands=[('generation_gpu4',[4],['collect','--shard','0','--shards','2']),('generation_gpu5',[5],['collect','--shard','1','--shards','2']),
              ('features',[6],['features','--follow']),('judge',[7],['judge','--follow'])]
    jobs=[]
    try:
        for name,gpus,args in commands:
            job=spawn(['-m','pgm_asc.collection.study',*args,'--config',path,'--split',split],gpus,Path(cfg['run_dir'])/'logs'/f'{split}_{name}.log');jobs.append((name,job))
        while True:
            if fit_worker[0].poll() not in [None,0]:raise RuntimeError('CPU fit worker failed; see fit_worker.log')
            for name,(p,h) in jobs:
                if p.poll() not in [None,0]:raise RuntimeError(f'{Path(cfg["run_dir"]).name}/{split}/{name} failed with exit {p.returncode}')
            update(protocol,f'{Path(cfg["run_dir"]).name}/{split}',workers={name:p.pid for name,(p,h) in jobs},fit_worker_pid=fit_worker[0].pid)
            if all(p.poll()==0 for name,(p,h) in jobs):break
            time.sleep(15)
        if not bank_complete(cfg,split):raise RuntimeError('Workers exited with incomplete bank')
    finally:
        for name,job in jobs:stop(job)


def evaluate_ready(protocol,fit_worker):
    from pgm_asc.evaluation.study import evaluate
    for path in protocol['jobs']:
        cfg=read_json(path);root=Path(cfg['run_dir'])
        if not (root/'selection_lock.json').exists():continue
        for split in cfg['evaluation_splits']:
            if (root/split/'study_evaluation.json').exists():continue
            run_bank(protocol,path,split,fit_worker);update(protocol,f'evaluate_{root.name}/{split}',fit_worker_pid=fit_worker[0].pid)
            evaluate(cfg,split);log('study cell split evaluated',cell=root.name,split=split)


def main():
    protocol=read_json('configs/pgm_followup.json');ROOT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(ROOT/'.supervisor.lock'),timeout=0):
        worker=spawn(['-m','pgm_asc.experiments.study_fit','--all'],[],ROOT/'fit_worker.log')
        try:
            for path in protocol['jobs']:
                cfg=read_json(path)
                if cfg.get('transfer_from'):continue
                for split in ['calibration','train','dev']:run_bank(protocol,path,split,worker)
                evaluate_ready(protocol,worker)
            while worker[0].poll() is None:
                evaluate_ready(protocol,worker);update(protocol,'waiting_for_remaining_fit_locks',fit_worker_pid=worker[0].pid);time.sleep(10)
            if worker[0].returncode!=0:raise RuntimeError('CPU fit worker failed')
            evaluate_ready(protocol,worker)
            for path in protocol['jobs']:
                cfg=read_json(path)
                if not all((Path(cfg['run_dir'])/s/'study_evaluation.json').exists() for s in cfg['evaluation_splits']):raise RuntimeError('Missing final evaluation '+path)
            update(protocol,'complete')
        except BaseException as e:
            write_json(ROOT/'failure.json',{'at':utc_now(),'error':repr(e),'traceback':traceback.format_exc()});update(protocol,'failed',error=repr(e));raise
        finally:stop(worker)


if __name__=='__main__':main()
