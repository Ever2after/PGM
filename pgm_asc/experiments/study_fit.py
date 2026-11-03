"""Structure ablations: equal labels/optimizer and common no-value stopping."""
import argparse,time,os
from pathlib import Path
import numpy as np
import torch
from pgm_asc.core.io import read_json,read_jsonl,write_json,write_jsonl,utc_now,log
from pgm_asc.core.integrity import load_tensor,file_sha,verify_lock,source_files
from pgm_asc.data.reasc_bank import load,build,traces
from pgm_asc.data.banks import fit_representation,fit_scalers,scale
from pgm_asc.models.partitions import unit
from pgm_asc.models.study import ARMS,fit,predict
from pgm_asc.policies.calibration import global_calibrator,calibrate
from pgm_asc.policies.stopping import GAMMAS,select_candidate
from pgm_asc.evaluation.replay import replay_policy
from pgm_asc.policies.reasc_reference import offline_calibrate,trace,exact_groups


def output_tokens(record):
    return record['generated_token_count']+sum(record[k]['output_tokens'] for k in ['extraction_inline','extraction_final'])


def ready(cfg):
    if cfg.get('transfer_from'):return (Path(cfg['transfer_from'])/'selection_lock.json').exists()
    root=Path(cfg['run_dir'])
    for split in ['train','dev','calibration']:
        for q in read_jsonl(Path(cfg['data_dir'])/f'{split}.jsonl'):
            qid=q['question_id']
            if not (root/split/'judgments'/f'{qid}.json').exists():return False
            if not all((root/split/'scored_responses'/f'{qid}__s{i:03d}.json').exists() for i in range(16)):return False
    return True


def attach(data,banks,meta,representation):
    vectors=np.zeros((len(banks),32,banks[0]['vectors'].shape[-1]),np.float32)
    for qi,b in enumerate(banks):
        raw=b['vectors'];v=unit(unit(raw)-np.asarray(representation['mean'],np.float32));v[np.linalg.norm(raw,axis=1)<=1e-8]=0;vectors[qi,:16]=v
        total=0
        for t,r in enumerate(b['records']):
            total+=output_tokens(r);row=meta[qi*16+t]
            if row['question_id']!=b['question']['question_id']:raise ValueError('Cohort order mismatch')
            row['reference_chat_tokens']=row['token_cost'];row['token_cost']=total
    data['embeddings']=torch.from_numpy(vectors);return data,meta


def datasets(cfg):
    root=Path(cfg['run_dir']);banks={s:load(cfg,s) for s in ['train','dev','calibration']}
    old=cfg['model_tag']=='gemma4b' and cfg['dataset']=='math'
    if old:
        parent=Path('artifacts/runs/reasc_pgm');verify_lock(read_json(parent/'selection_lock.json'))
        rep=read_json(parent/'representation.json');scalers=read_json(parent/'scalers.json')
    else:
        rep=fit_representation(banks['train']);scalers=None
    data={};metadata={}
    for split in banks:
        if old:
            d=load_tensor(parent/'cache'/f'{split}.pt');m=read_jsonl(parent/'cache'/f'{split}_metadata.jsonl')
        else:
            d,m=build(banks[split],rep)
            if split=='train':scalers=fit_scalers(d)
            d=scale(d,scalers)
        data[split],metadata[split]=attach(d,banks[split],m,rep)
    write_json(root/'representation.json',rep);write_json(root/'scalers.json',scalers)
    return banks,data,metadata


def sc_accuracy(banks):
    correct=[]
    for b in banks:
        _,ids=exact_groups([r['prediction'] for r in b['records']]);c=int(np.bincount(ids).argmax());correct.append(b['labels']['canonical_correct'][c])
    return float(np.mean(correct))


def native_decision(bank,method,settings=None,beta=.95):
    pred=[r['prediction'] for r in bank['records']];scores=[r['bottom10_confidence'] for r in bank['records']]
    h=trace(pred,scores,'reasc' if method=='reasc' else 'asc',settings)
    if method=='reasc' and scores[0]>=settings['gate']:return 0,int(h['chosen'][0]),'single_gate'
    hits=np.flatnonzero(h['confidence']>=beta) if beta is not None else []
    i=int(hits[0]) if len(hits) else 15
    return i,int(h['chosen'][i]),'confidence' if len(hits) else 'cap'


def tune_native(banks,settings,sc):
    result={}
    for method in ['asc','reasc']:
        candidates=[]
        for gamma in GAMMAS:
            acc=[];tokens=[];samples=[]
            for b in banks:
                i,c,_=native_decision(b,method,settings,gamma);acc.append(b['labels']['canonical_correct'][c]);samples.append(i+1);tokens.append(sum(output_tokens(r) for r in b['records'][:i+1]))
            candidates.append({'gamma':gamma,'accuracy':float(np.mean(acc)),'mean_tokens':float(np.mean(tokens)),'mean_samples':float(np.mean(samples))})
        chosen,_=select_candidate(candidates,sc);result[method]={'selected':chosen,'candidates':candidates}
    return result


def select(cfg,config_path):
    root=Path(cfg['run_dir']);root.mkdir(parents=True,exist_ok=True);path=root/'selection_lock.json'
    if path.exists():verify_lock(read_json(path));return
    started=time.perf_counter();torch.set_num_threads(4)
    if cfg.get('transfer_from'):
        source=Path(cfg['transfer_from'])/'selection_lock.json';lock=read_json(source);verify_lock(lock)
        result={k:lock[k] for k in ['methods','reasc_offline','native_tuned','preprocessing','train_seeds','primary_seed']}
        result.update(created_at=utc_now(),dataset='omnimath',model_tag=cfg['model_tag'],transfer_from=str(source),holdout_accessed=False,
                      artifact_sha256=lock['artifact_sha256']|{str(source):file_sha(source),config_path:file_sha(config_path)},
                      transfer_note='No Omni fitting, calibration, threshold selection or labels used. Every method and baseline offline calibration transferred from this model MATH cell.')
        write_json(path,result);return
    banks,data,meta=datasets(cfg);sc=sc_accuracy(banks['dev'])
    offline=offline_calibrate([b['records'][0]['lowest_confidence'] for b in banks['calibration']],
                              [b['labels']['raw_response_correct'][0] for b in banks['calibration']])
    methods={};progress=[]
    for seed in cfg['train_seeds']:
        for name,spec in ARMS.items():
            out=root/'models'/name/f'seed{seed}';model=fit(data['train'],data['dev'],spec,cfg,out,seed)
            cal=global_calibrator(predict(model,data['calibration']),data['calibration'])
            histories=traces(calibrate(predict(model,data['dev']),data['dev']['t'],cal),data['dev'])
            candidates=[]
            for gamma in GAMMAS:
                policy={'minimum':2,'patience':1,'cap':16,'gamma':gamma}
                r=replay_policy(histories,data['dev'],meta['dev'],policy)
                candidates.append({k:v for k,v in r.items() if k!='rows'}|{'policy':policy})
            chosen,_=select_candidate(candidates,sc)
            entry={'checkpoint':str(out/'checkpoint.pt'),'spec':spec,'seed':seed,'calibrator':cal,'selected':chosen,
                   'fixed95':{'minimum':2,'patience':1,'cap':16,'gamma':.95},'candidates':candidates}
            methods[f'{name}_s{seed}']=entry;write_json(out/'selection.json',entry)
            progress.append({'arm':name,'seed':seed,'dev_accuracy':chosen['accuracy'],'dev_output_tokens':chosen['mean_tokens'],'feasible':chosen['feasible']})
            write_json(root/'fit_status.json',{'state':'running','completed_fits':len(progress),'total_fits':len(ARMS)*len(cfg['train_seeds']),'latest':progress[-1],'updated_at':utc_now()})
        # Same-head intervention separates partition choice from retraining effects.
        src=methods[f'mixture_s{seed}'];model=fit(data['train'],data['dev'],ARMS['mixture'],cfg,root/'models/mixture'/f'seed{seed}',seed)
        model.mode='single';model.partition=1
        out=root/'models/same_head_single'/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':model.state_dict(),'spec':{'mode':'single','partition':1},'embedding_dim':data['train']['embeddings'].shape[-1],'seed':seed},out/'checkpoint.pt')
        cal=global_calibrator(predict(model,data['calibration']),data['calibration'])
        h=traces(calibrate(predict(model,data['dev']),data['dev']['t'],cal),data['dev']);candidates=[]
        for gamma in GAMMAS:
            policy={'minimum':2,'patience':1,'cap':16,'gamma':gamma};r=replay_policy(h,data['dev'],meta['dev'],policy)
            candidates.append({k:v for k,v in r.items() if k!='rows'}|{'policy':policy})
        chosen,_=select_candidate(candidates,sc)
        methods[f'same_head_single_s{seed}']={'checkpoint':str(out/'checkpoint.pt'),'seed':seed,'calibrator':cal,'selected':chosen,'fixed95':src['fixed95'],'intervention_from':src['checkpoint']}
    write_json(root/'fit_status.json',{'state':'complete','completed_fits':len(progress),'total_fits':len(progress),'updated_at':utc_now()})
    native=tune_native(banks['dev'],offline,sc)
    files=source_files()+[Path(config_path),Path('configs/pgm_followup.json'),Path('docs/PGM_FOLLOWUP_PLAN.md'),Path('artifacts/data/pgm_followup/manifest.json')]
    files+=list(Path(cfg['data_dir']).glob('*.jsonl'))+[root/'representation.json',root/'scalers.json']
    files+=list((root/'models').rglob('checkpoint.pt'))+list((root/'models').rglob('selection.json'))+list((root/'models').rglob('fit.json'))
    lock={'created_at':utc_now(),'holdout_accessed':False,'dataset':cfg['dataset'],'model_tag':cfg['model_tag'],'methods':methods,'reasc_offline':offline,
          'native_tuned':native,'dev_sc16_accuracy':sc,'train_seeds':cfg['train_seeds'],'primary_seed':42,'selection_objective':'total output tokens, both extraction passes included',
          'preprocessing':{'representation':str(root/'representation.json'),'scalers':str(root/'scalers.json')},'selection_seconds':time.perf_counter()-started,
          'artifact_sha256':{str(p):file_sha(p) for p in sorted(set(files)) if p.is_file()}}
    write_json(path,lock);log('study heads and policies locked',cell=root.name,heads=len(methods),seconds=lock['selection_seconds'])


def worker(protocol):
    for config_path in protocol['jobs']:
        cfg=read_json(config_path)
        while not ready(cfg):time.sleep(10)
        select(cfg,config_path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--all',action='store_true');a=p.parse_args()
    if a.all:worker(read_json('configs/pgm_followup.json'))
    else:select(read_json(a.config),a.config)
