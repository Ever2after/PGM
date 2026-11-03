"""User-scoped Qwen7B queues; two generators share one TF/judge service."""
import argparse,os,signal,subprocess,time,traceback
from pathlib import Path
from filelock import FileLock
from pgm_asc.core.io import read_json,write_json,utc_now,log
from pgm_asc.core.integrity import verify_lock
from pgm_asc.experiments import study_runner as base
from pgm_asc.experiments.study_runner_health import new_traceback

ROOT=Path('artifacts/runs/pgm_followup')
SCOPE='configs/pgm_followup_runtime_scope.json'


def cached_loader(factory):
    cache={}
    def load(*args,**kwargs):
        key=str(args[0])
        if cache and key not in cache:raise ValueError('A shared model service cannot change checkpoints')
        if key not in cache:cache[key]=factory(*args,**kwargs)
        return cache[key]
    return load


def service(kind):
    import pgm_asc.collection.study as collection
    scope=read_json(SCOPE);configs=[read_json(p) for p in scope['active_jobs']]
    assert len({(c['model_id'],c['model_revision']) for c in configs})==1
    if kind=='features':
        from transformers import AutoModelForCausalLM,AutoTokenizer
        model_load=cached_loader(AutoModelForCausalLM.from_pretrained)
        tokenizer_load=cached_loader(AutoTokenizer.from_pretrained)
        AutoModelForCausalLM.from_pretrained=classmethod(lambda cls,*a,**kw:model_load(*a,**kw))
        AutoTokenizer.from_pretrained=classmethod(lambda cls,*a,**kw:tokenizer_load(*a,**kw))
    else:
        original=collection.initialize_judge;cache=[]
        def initialize(cfg):
            if not cache:cache.append(original(cfg))
            return cache[0]
        collection.initialize_judge=initialize
    with FileLock(str(ROOT/f'.qwen7_{kind}.lock'),timeout=0):
        while not (ROOT/'qwen7_only_complete.json').exists():
            for cfg in configs:
                for split in ['calibration','train','dev']+cfg['evaluation_splits']:
                    out=Path(cfg['run_dir'])/split
                    if not any((out/'questions').glob('*.json')):continue
                    if split not in ['calibration','train','dev'] and not (Path(cfg['run_dir'])/'selection_lock.json').exists():continue
                    if kind=='features':collection.features(cfg,split,follow=False)
                    else:collection.judge(cfg,split,follow=False)
            write_json(ROOT/f'qwen7_{kind}_service.json',{'state':'running','pid':os.getpid(),'updated_at':utc_now(),'shared_checkpoint':True})
            time.sleep(3)


def command(args,gpus,logpath):
    offset=logpath.stat().st_size if logpath.exists() else 0
    job=base.spawn(args,gpus,logpath)
    try:
        while job[0].poll() is None:
            if new_traceback(logpath,offset):raise RuntimeError('New fatal traceback: '+str(logpath))
            time.sleep(3)
        if job[0].returncode:raise RuntimeError(f'Exit {job[0].returncode}: {logpath}')
    finally:base.stop(job)


def cell(config_path):
    cfg=read_json(config_path);root=Path(cfg['run_dir']);gpu=read_json(SCOPE)['generation_gpu_by_dataset'][cfg['dataset']]
    def status(phase):write_json(root/'queue_status.json',{'phase':phase,'pid':os.getpid(),'generation_gpu':gpu,'updated_at':utc_now()})
    def bank(split):
        if base.bank_complete(cfg,split):return
        status('collect_'+split)
        command(['-m','pgm_asc.collection.study','collect','--config',config_path,'--split',split,'--shard','0','--shards','1'],[gpu],root/'logs'/f'{split}_single_gpu{gpu}.log')
        status('features_judge_'+split)
        while not base.bank_complete(cfg,split):time.sleep(3)
    with FileLock(str(root/'.qwen7_queue.lock'),timeout=0):
        for split in ['calibration','train','dev']:bank(split)
        status('fit_and_lock')
        command(['-m','pgm_asc.experiments.study_fit','--config',config_path],[],root/'fit_queue.log')
        verify_lock(read_json(root/'selection_lock.json'))
        for split in cfg['evaluation_splits']:
            if (root/split/'study_evaluation.json').exists():continue
            bank(split);status('evaluate_'+split)
            # Keep CPU evaluation isolated from the persistent service processes.
            command(['-m','pgm_asc.experiments.qwen7_only','evaluate','--config',config_path,'--split',split],[],root/'logs'/f'{split}_evaluate.log')
        status('complete')


def main():
    scope=read_json(SCOPE);protocol=read_json('configs/pgm_followup.json');jobs=[]
    with FileLock(str(ROOT/'.supervisor.lock'),timeout=0):
        try:
            assert not (ROOT/'qwen7_only_complete.json').exists()
            for kind,gpu in [('features',6),('judge',7)]:
                path=ROOT/f'qwen7_shared_{kind}.log';offset=path.stat().st_size if path.exists() else 0
                jobs.append((kind,base.spawn(['-m','pgm_asc.experiments.qwen7_only',kind],[gpu],path),path,offset))
            for path in scope['active_jobs']:
                cfg=read_json(path);lp=Path(cfg['run_dir'])/'queue.log';offset=lp.stat().st_size if lp.exists() else 0
                jobs.append((cfg['dataset'],base.spawn(['-m','pgm_asc.experiments.qwen7_only','cell','--config',path],[],lp),lp,offset))
            while True:
                states={}
                for path in scope['active_jobs']:
                    cfg=read_json(path);sp=Path(cfg['run_dir'])/'queue_status.json'
                    states[cfg['dataset']]=read_json(sp) if sp.exists() else {'phase':'starting'}
                for name,(p,h),lp,offset in jobs:
                    if new_traceback(lp,offset):raise RuntimeError('New fatal traceback: '+str(lp))
                    if p.poll() not in [None,0]:raise RuntimeError(f'{name} exited {p.returncode}: {lp}')
                complete=all(x['phase']=='complete' for x in states.values())
                base.update(protocol,'qwen7_complete_omni_deferred' if complete else 'qwen7_parallel',
                            active_jobs=scope['active_jobs'],deferred_jobs=scope['deferred_jobs'],queue_states=states,
                            workers={name:p.pid for name,(p,h),lp,offset in jobs})
                with (ROOT/'progress.md').open('a') as f:
                    f.write('\nOmni-MATH: **deferred at user request**. Active queues: Qwen7B MATH on GPU4; GSM8K on GPU5. GPU6/7 are shared feature/judge services.\n')
                    for name,state in states.items():f.write(f"\n- {name}: {state['phase']}\n")
                if complete:break
                time.sleep(10)
            write_json(ROOT/'qwen7_only_complete.json',{'completed_at':utc_now(),'active_jobs':scope['active_jobs'],'omni_deferred':True})
        except BaseException as e:
            write_json(ROOT/'failure.json',{'at':utc_now(),'error':repr(e),'traceback':traceback.format_exc()});base.update(protocol,'failed',error=repr(e),deferred_jobs=scope['deferred_jobs']);raise
        finally:
            # A cell may have an independent generation/evaluation child session.
            # Normal completion exits those first; failure cleanup includes descendants.
            for name,job,lp,offset in jobs:
                if name in ['math','gsm8k'] and job[0].poll() is None:
                    for path in Path('/proc').glob('[0-9]*/stat'):
                        try:
                            raw=path.read_text();fields=raw[raw.rfind(')')+2:].split()
                            if int(fields[1])==job[0].pid:
                                pid=int(path.parent.name)
                                try:os.killpg(pid,signal.SIGTERM)
                                except ProcessLookupError:pass
                        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
                base.stop(job)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',nargs='?',default='run',choices=['run','cell','features','judge','evaluate']);p.add_argument('--config');p.add_argument('--split');a=p.parse_args()
    if a.command=='run':main()
    elif a.command=='cell':cell(a.config)
    elif a.command=='evaluate':
        from pgm_asc.evaluation.study import evaluate
        evaluate(read_json(a.config),a.split)
    else:service(a.command)
