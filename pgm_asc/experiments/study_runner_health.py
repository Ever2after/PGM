"""Runtime-only health checks; keep already-locked study modules untouched."""
import time
from pathlib import Path
from pgm_asc.core.io import read_json
from pgm_asc.core.integrity import verify_lock
from pgm_asc.experiments import study_runner as base


def new_traceback(path,offset):
    path=Path(path)
    if not path.exists():return False
    with path.open('rb') as f:
        f.seek(offset)
        return b'Traceback (most recent call last)' in f.read()


def run_bank(protocol,path,split,fit_worker):
    cfg=read_json(path)
    if base.bank_complete(cfg,split):return
    if (Path(cfg['run_dir'])/split).is_symlink():raise ValueError('Never write linked original bank')
    if split not in ['train','dev','calibration']:verify_lock(read_json(Path(cfg['run_dir'])/'selection_lock.json'))
    commands=[('generation_gpu4',[4],['collect','--shard','0','--shards','2']),('generation_gpu5',[5],['collect','--shard','1','--shards','2']),
              ('features',[6],['features','--follow']),('judge',[7],['judge','--follow'])]
    jobs=[]
    try:
        for name,gpus,args in commands:
            logpath=Path(cfg['run_dir'])/'logs'/f'{split}_{name}.log'
            offset=logpath.stat().st_size if logpath.exists() else 0
            job=base.spawn(['-m','pgm_asc.collection.study',*args,'--config',path,'--split',split],gpus,logpath)
            jobs.append((name,job,logpath,offset))
        while True:
            if fit_worker[0].poll() not in [None,0]:raise RuntimeError('CPU fit worker failed; see fit_worker.log')
            for name,(p,h),logpath,offset in jobs:
                if new_traceback(logpath,offset):raise RuntimeError(f'{cfg["run_dir"]}/{split}/{name}: traceback detected even if engine teardown has not exited; see {logpath}')
                if p.poll() not in [None,0]:raise RuntimeError(f'{cfg["run_dir"]}/{split}/{name} exited {p.returncode}')
            base.update(protocol,f'{Path(cfg["run_dir"]).name}/{split}',workers={name:p.pid for name,(p,h),logpath,offset in jobs},fit_worker_pid=fit_worker[0].pid,health_checks='new-segment traceback and exit status')
            if all(p.poll()==0 for name,(p,h),logpath,offset in jobs):break
            time.sleep(10)
        if not base.bank_complete(cfg,split):raise RuntimeError('Workers exited with incomplete bank')
    finally:
        for name,job,logpath,offset in jobs:base.stop(job)


def main():
    base.run_bank=run_bank
    # Resume the interrupted fitting-data stage before dispatching ready tests.
    evaluate_ready=base.evaluate_ready
    def resume_order(protocol,worker):
        cfg=read_json('configs/pgm_followup/qwen3b_math.json')
        if not base.bank_complete(cfg,'dev'):return
        return evaluate_ready(protocol,worker)
    base.evaluate_ready=resume_order
    base.main()


if __name__=='__main__':main()
