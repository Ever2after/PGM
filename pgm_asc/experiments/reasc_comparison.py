"""Train/calibrate/select on train/dev/cal only; immutable test policy lock."""
from __future__ import annotations
import argparse,copy,time
from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from pgm_asc.core.io import read_json,write_json,write_jsonl,utc_now,log
from pgm_asc.core.integrity import file_sha,load_tensor,verify_lock,source_files
from pgm_asc.data.reasc_bank import load,build,traces
from pgm_asc.data.banks import fit_representation,fit_scalers,scale
from pgm_asc.models.selector import PGM,predict,batch_inputs
from pgm_asc.policies.calibration import global_calibrator,calibrate
from pgm_asc.policies.continuation import continuation_predict
from pgm_asc.policies.stopping import GAMMAS,select_candidate
from pgm_asc.policies.reasc_reference import offline_calibrate
from pgm_asc.training.continuation import fit_continuation
from pgm_asc.evaluation.replay import replay_policy


def prefix_mass():
    return np.array([.2,.2]+[.1]*2+[.05]*4+[.025]*8,dtype=np.float32)


def fit(train,dev,spec,cfg,root,seed=42,epochs=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'result.json').exists():return restore(root/'checkpoint.pt')
    p=cfg['train'];torch.manual_seed(seed);torch.set_num_threads(4)
    model=PGM(32,**spec);optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    weights=torch.tensor(prefix_mass());rng=np.random.default_rng(seed);qcount=len(train['t'])//16
    best=float('inf');bad=0;state=None;history=[];started=time.perf_counter()
    for epoch in range(epochs or p['epochs']):
        model.train();order=rng.permutation(qcount);losses=[]
        for start in range(0,qcount,p['question_batch']):
            idx=torch.from_numpy((order[start:start+p['question_batch'],None]*16+np.arange(16)).reshape(-1))
            lp=model(batch_inputs(train,idx,32));loss=(F.nll_loss(lp,train['target'][idx],reduction='none').reshape(-1,16)*weights).sum(1).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError('Nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        lp,_=predict(model,dev);target=dev['target'].numpy();criterion=float((-lp[np.arange(len(lp)),target].reshape(-1,16)*prefix_mass()).sum(1).mean())
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'dev_five_bucket_nll':criterion})
        if criterion<best-1e-7:best=criterion;bad=0;best_epoch=epoch+1;state=copy.deepcopy(model.state_dict())
        else:bad+=1
        write_json(root/'progress.json',history[-1]|{'best_epoch':best_epoch,'updated_at':utc_now()})
        if epoch==0 or (epoch+1)%10==0:log('reference-setting PGM fit',mode=spec['mode'],epoch=epoch+1,dev_nll=criterion)
        if bad>=p['patience']:break
    model.load_state_dict(state);torch.save({'state_dict':state,'spec':spec,'n':32,'seed':seed},root/'checkpoint.pt')
    write_jsonl(root/'history.jsonl',history);write_json(root/'result.json',{'epochs':len(history),'best_epoch':best_epoch,'dev_nll':best,'seed':seed,'spec':spec,'elapsed_seconds':time.perf_counter()-started,'objective':'five equal-mass prefix buckets 1/2/3-4/5-8/9-16','alpha':float(model.alpha.detach())})
    return model


def restore(path):
    saved=load_tensor(path);model=PGM(saved['n'],**saved['spec']);model.load_state_dict(saved['state_dict']);model.eval();return model


def frozen_preprocessing():
    cfg=read_json('configs/experiment.json')
    # Locate the original bootstrap cache without depending on moved training outputs.
    old=Path('artifacts/runs/pgm_asc/bootstrap/cache')
    return read_json(old/'representation.json'),read_json(old/'scalers.json')


def setup_costs(banks):
    def count(records):
        return {'responses':len(records),'reference_chat_tokens':sum(r['reference_chat_tokens'] for r in records),
                'generated_tokens':sum(r['generated_token_count'] for r in records),
                'extraction_input_tokens':sum(r[k]['input_tokens'] for r in records for k in ['extraction_inline','extraction_final']),
                'extraction_output_tokens':sum(r[k]['output_tokens'] for r in records for k in ['extraction_inline','extraction_final']),
                'feature_input_tokens':sum(r['feature_input_tokens'] for r in records),
                'generation_seconds_share':sum(r['native_generation_seconds_share'] for r in records),
                'extraction_seconds_share':sum(r[k]['seconds_share'] for r in records for k in ['extraction_inline','extraction_final']),
                'feature_seconds_share':sum(r['feature_seconds_share'] for r in records)}
    per_split={s:count([r for b in bs for r in b['records']]) for s,bs in banks.items()}
    first=count([b['records'][0] for b in banks['calibration']]);first['feature_input_tokens']=0;first['feature_seconds_share']=0.
    return {'pgm_collection_by_split':per_split,'reasc_offline_first_response_calibration':first,
            'pgm_setup_judge_seconds':sum(b['labels']['elapsed_seconds'] for bs in banks.values() for b in bs),
            'pgm_setup_judge_input_tokens':sum(b['labels']['input_tokens'] for bs in banks.values() for b in bs),
            'note':'PGM setup includes all train/dev/cal responses and both label profiles. Collection shares are offline accounting, not causal serving latency. Selector random-control fit is excluded from primary mixture fitting time but included in total selection wall time.'}


def select(cfg):
    root=Path(cfg['run_dir']);lockpath=root/'selection_lock.json'
    if lockpath.exists():verify_lock(read_json(lockpath));return
    started=time.perf_counter();banks={s:load(cfg,s) for s in ['train','dev','calibration']}
    representation=fit_representation(banks['train']);write_json(root/'representation.json',representation)
    raw={s:build(b,representation) for s,b in banks.items()};scalers=fit_scalers(raw['train'][0]);write_json(root/'scalers.json',scalers)
    data={s:scale(d[0],scalers) for s,d in raw.items()};meta={s:d[1] for s,d in raw.items()}
    for s in data:
        (root/'cache').mkdir(exist_ok=True);torch.save(data[s],root/'cache'/f'{s}.pt');write_jsonl(root/'cache'/f'{s}_metadata.jsonl',meta[s])
    offline=offline_calibrate([b['records'][0]['lowest_confidence'] for b in banks['calibration']],
                              [b['labels']['raw_response_correct'][0] for b in banks['calibration']])
    write_json(root/'reasc_offline_calibration.json',offline)
    mixture=fit(data['train'],data['dev'],{'mode':'mixture'},cfg,root/'models/mixture')
    random=fit(data['train'],data['dev'],{'mode':'random'},cfg,root/'models/random')
    single=PGM(32,mode='single',partition=1);single.load_state_dict(mixture.state_dict())
    (root/'models/single').mkdir(parents=True,exist_ok=True);torch.save({'state_dict':single.state_dict(),'spec':{'mode':'single','partition':1},'n':32,'seed':42},root/'models/single/checkpoint.pt')
    dev_sc=[]
    for b in banks['dev']:
        from pgm_asc.policies.reasc_reference import exact_groups
        _,ids=exact_groups([r['prediction'] for r in b['records']]);dev_sc.append(b['labels']['canonical_correct'][int(np.bincount(ids).argmax())])
    sc=float(np.mean(dev_sc));methods={}
    for name,model in [('mixture',mixture),('random',random),('single',single)]:
        mroot=root/'models'/name
        lp={s:predict(model,data[s])[0] for s in data};calibrator=global_calibrator(lp['calibration'],data['calibration'])
        write_json(mroot/'calibrator.json',calibrator)
        histories={s:traces(calibrate(lp[s],data[s]['t'],calibrator),data[s]) for s in data}
        all_candidates=[]
        for cap in cfg['caps']:
            croot=mroot/f'continuation_cap{cap}';croot.mkdir(exist_ok=True)
            fitted,_=fit_continuation(histories['train'],histories['dev'],data['train'],data['dev'],meta['train'],meta['dev'],croot,cap=cap)
            values=continuation_predict(fitted,histories['dev'],data['dev'])
            for gamma in GAMMAS:
                for lam in cfg['lambdas']:
                    policy={'minimum':2,'patience':1,'cap':cap,'gamma':gamma,'continuation_lambda':lam}
                    result=replay_policy(histories['dev'],data['dev'],meta['dev'],policy,values)
                    all_candidates.append({k:v for k,v in result.items() if k!='rows'}|{'policy':policy,'continuation_head':str(croot/'continuation.joblib')})
        chosen,_=select_candidate(all_candidates,sc);write_json(mroot/'policy_search.json',{'reference_sc16_accuracy':sc,'selected':chosen,'candidates':all_candidates})
        methods['pgm_'+name]={'checkpoint':str(mroot/'checkpoint.pt'),'calibrator':calibrator,**chosen}
        if name=='mixture':
            fixed={'minimum':2,'patience':1,'cap':16,'gamma':.95}
            methods['pgm_fixed95']={'checkpoint':str(mroot/'checkpoint.pt'),'calibrator':calibrator,'policy':fixed}
            no_value=[]
            for gamma in GAMMAS:
                policy=fixed|{'gamma':gamma};r=replay_policy(histories['dev'],data['dev'],meta['dev'],policy)
                no_value.append({k:v for k,v in r.items() if k!='rows'}|{'policy':policy})
            nv,_=select_candidate(no_value,sc);methods['pgm_no_value']={'checkpoint':str(mroot/'checkpoint.pt'),'calibrator':calibrator,**nv}
            write_json(mroot/'no_value_search.json',no_value)
    oldpath=Path('artifacts/runs/pgm_asc/selector/selection_lock.json');oldlock=read_json(oldpath);verify_lock(oldlock)
    old=oldlock['stages']['s6_continuation']['mixture']['selected']
    methods['frozen_v2_transfer']={'checkpoint':'artifacts/runs/pgm_asc/selector/refits/mixture/checkpoint.pt',**{k:old[k] for k in ['calibrator','policy','continuation_head']},'transfer_note':'Old preprocessing, T and policy; length16 bank, original selected cap8 unchanged; exact reference grouping adapter.'}
    frozen_rep,frozen_scalers=frozen_preprocessing()
    write_json(root/'frozen_representation.json',frozen_rep);write_json(root/'frozen_scalers.json',frozen_scalers)
    lock={'created_at':utc_now(),'holdout_accessed':False,'methods':methods,'dev_sc16_accuracy':sc,'supervision_questions':{'train':len(banks['train']),'dev':len(banks['dev']),'calibration':len(banks['calibration'])},
          'reference_train_free_calibration_questions':128,'extra_supervision_relative_reasc':1000,'elapsed_seconds':time.perf_counter()-started,'artifact_sha256':{}}
    lock['setup_costs']=setup_costs(banks)
    lock['setup_costs']['primary_mixture_fit_seconds']=read_json(root/'models/mixture/result.json')['elapsed_seconds']
    files=source_files()+list(Path('configs').glob('reasc_comparison.json'))+list(Path(cfg['data_dir']).glob('*'))
    files += [Path('third_party/reasc_sources.json'),Path('docs/REASC_PGM_COMPARISON_PLAN.md')]
    files += list(Path('third_party/xverify_official/src/xVerify/prompts').glob('*.py'))
    immutable=['representation.json','scalers.json','frozen_representation.json','frozen_scalers.json',
               'reasc_offline_calibration.json','smoke_validation.json','reference_validation.json',
               'environment.json','requirements.lock.txt','runtime_migration.json',
               'runtime_optimization_start.json','runtime_config_before_optimization.json','runtime_benchmark.json']
    files += [root/name for name in immutable if (root/name).exists()]
    files += list((root/'models').rglob('*'))
    for split in ['train','dev','calibration']:
        files += list((root/split).glob('generation_manifest.json'))
        files += list((root/split/'generation_manifests').glob('*.json'))
    # Status/log/progress may change; lock only durable learned artifacts and protocol sources.
    files=[p for p in files if p.is_file() and p.name not in ['status.json','reference_validation.json','progress.json','selection_lock.json']]
    files+=[Path(m['checkpoint']) for m in methods.values()]+[Path(old['continuation_head']),oldpath]
    lock['artifact_sha256']={str(p):file_sha(p) for p in sorted(set(files))}
    write_json(lockpath,lock);log('New policies frozen before test generation',methods=len(methods),elapsed_seconds=lock['elapsed_seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/reasc_comparison.json');a=p.parse_args();select(read_json(a.config))
