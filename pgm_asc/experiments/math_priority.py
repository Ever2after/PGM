"""Complete locked Qwen7B MATH first on GPU4-7, then resume GSM8K."""
import time,traceback
from pathlib import Path
from filelock import FileLock
from pgm_asc.core.io import read_json,write_json,utc_now
from pgm_asc.core.integrity import verify_lock
from pgm_asc.experiments import study_runner as base
from pgm_asc.experiments.study_runner_health import new_traceback

ROOT=Path('artifacts/runs/pgm_followup')


def main():
    protocol=read_json('configs/pgm_followup.json');scope=read_json('configs/pgm_followup_runtime_scope.json')
    jobs=[]
    with FileLock(str(ROOT/'.supervisor.lock'),timeout=0):
        try:
            for dataset,split in [('math','confirm'),('gsm8k','test')]:
                path=f'configs/pgm_followup/qwen7b_{dataset}.json';cfg=read_json(path);root=Path(cfg['run_dir'])
                verify_lock(read_json(root/'selection_lock.json'))
                if not base.bank_complete(cfg,split):
                    commands=[('generation_gpu4',[4],['collect','--shard','0','--shards','2']),
                              ('generation_gpu5',[5],['collect','--shard','1','--shards','2']),
                              ('features',[6],['features','--follow']),('judge',[7],['judge','--follow'])]
                    for name,gpus,args in commands:
                        lp=root/'logs'/f'{split}_priority_{name}.log';offset=lp.stat().st_size if lp.exists() else 0
                        jobs.append((name,base.spawn(['-m','pgm_asc.collection.study',*args,'--config',path,'--split',split],gpus,lp),lp,offset))
                    while True:
                        for name,(p,h),lp,offset in jobs:
                            if new_traceback(lp,offset):raise RuntimeError('Fatal traceback: '+str(lp))
                            if p.poll() not in [None,0]:raise RuntimeError(f'{name} exited {p.returncode}: {lp}')
                        states={'math':{'phase':'collect_confirm' if dataset=='math' else 'complete'},'gsm8k':{'phase':'paused_for_math' if dataset=='math' else 'collect_test'}}
                        base.update(protocol,f'qwen7_{dataset}_priority',active_jobs=[path],deferred_jobs=scope['deferred_jobs'],queue_states=states,
                                    workers={name:p.pid for name,(p,h),lp,offset in jobs})
                        with (ROOT/'progress.md').open('a') as f:f.write(f'\nPriority: **{dataset}**; both GPU4/5 generate this dataset, GPU6 features, GPU7 judge. Omni deferred. GSM resumes after MATH completes.\n')
                        if all(p.poll()==0 for name,(p,h),lp,offset in jobs):break
                        time.sleep(10)
                    for name,job,lp,offset in jobs:base.stop(job)
                    jobs=[]
                    if not base.bank_complete(cfg,split):raise RuntimeError('Incomplete bank '+dataset)
                base.update(protocol,f'evaluate_qwen7_{dataset}',deferred_jobs=scope['deferred_jobs'])
                from pgm_asc.evaluation.study import evaluate
                evaluate(cfg,split)
                if dataset=='math':write_json(ROOT/'math_priority_complete.json',{'completed_at':utc_now(),'math500_evaluated':True,'confirm500_evaluated':True,'gsm_resumes_after_this':True})
            write_json(ROOT/'qwen7_only_complete.json',{'completed_at':utc_now(),'omni_deferred':True})
            base.update(protocol,'qwen7_complete_omni_deferred',deferred_jobs=scope['deferred_jobs'])
            with (ROOT/'progress.md').open('a') as f:f.write('\nQwen7B MATH and GSM8K complete. Omni-MATH remains deferred at user request.\n')
        except BaseException as e:
            write_json(ROOT/'failure.json',{'at':utc_now(),'error':repr(e),'traceback':traceback.format_exc()});base.update(protocol,'failed',error=repr(e));raise
        finally:
            for name,job,lp,offset in jobs:base.stop(job)


if __name__=='__main__':main()
