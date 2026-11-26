"""Create resumable P1 stopping traces; publish via the standard v2 plugin."""
from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from pgm_asc import config
from pgm_asc.p1_stopping.model import CountPosterior,trace
from pgm_asc.utils import digest,file_sha256,read_json,utc_now,write_json
from pgm_asc.v2 import CONFIG as V2

CONFIG=read_json(config.ROOT/'configs/p1_stopping.json')


def definitions(study=None):
    s=study or CONFIG;out={}
    for v in s['variants']:
        short=v.removeprefix('P1_')
        for m in s['training_methods']:
            suffix={'joint':'joint','mixed_uniform':'mixed','mixed_v1':'mixed_v1'}[m]
            name=f'p1_{short}_{suffix}'
            out[name]=dict(variant=v,training=m,stop='voi',label=f'P1 {short} '+{'joint':'Joint','mixed_uniform':'Mixed','mixed_v1':'Mixed v1'}[m])
            if m in s['confidence_controls']:
                out[name+'_confidence']={**out[name],'stop':'confidence','label':out[name]['label']+' confidence'}
    return out


def code_hashes():
    names=['p1_stopping/model.py','p1_stopping/run.py','p1_stopping/plugin.py',
        'hierarchical/model.py','v2/model.py','v2/voi.py','v2/run.py','v2/methods.py','v2/baselines.py']
    return {p:file_sha256(config.ROOT/'pgm_asc'/p) for p in names}


def settings(study=None):
    return dict(study=copy.deepcopy(study or CONFIG),v2=copy.deepcopy(V2),code_sha256=code_hashes())


def source_dir(study):
    p=Path(study['source_dir']);return p if p.is_absolute() else config.ROOT/p


def out_dir():return config.RESULTS/'p1_stopping'


def check_source(model,dataset,study):
    root=source_dir(study);source=root/'selection'/f'{model}_{dataset}.json'
    fitted=read_json(source);protocol=read_json(root/'protocol.json')
    # Old protocols express paths relative to pgm_asc; preserve them byte for byte.
    for rel,sha in protocol['code_sha256'].items():
        path=config.ROOT/'pgm_asc'/rel
        if not path.exists():path=config.ROOT/rel
        if file_sha256(path)!=sha:raise ValueError(f'Frozen P1 source changed: {rel}')
    bank=config.ARTIFACTS/'v2'/f'{model}_{dataset}.npz'
    if file_sha256(bank)!=fitted['bank_sha256']:raise ValueError('P1 fit and stopping bank differ')
    for d in definitions(study).values():
        name=f"{d['variant']}__{d['training']}"
        if name not in fitted['selected']:raise ValueError(f'Missing frozen head {name}')
    return fitted,source,bank


def cell(model,dataset,study,experiment_id,overwrite=False):
    torch.set_num_threads(1)
    started=time.perf_counter();root=out_dir();name=f'{model}_{dataset}'
    fitted,source,bank=check_source(model,dataset,study)
    source_sha=file_sha256(source)
    folder=config.ARTIFACTS/'p1_stopping'/experiment_id
    path=root/'cells'/f'{name}.json';cache=folder/'traces'/f'{name}.npz'
    if path.exists() and not overwrite:
        old=read_json(path)
        if old['experiment_id']!=experiment_id or old['source_fit_sha256']!=source_sha:raise ValueError('Existing P1 stopping provenance differs')
        if file_sha256(old['trace_path'])!=old['trace_sha256']:raise ValueError('Stopping trace changed')
        return str(path)
    lock=dict(experiment_id=experiment_id,model=model,dataset=dataset,source_fit_path=str(source),
        source_fit_sha256=source_sha,source_experiment_id=fitted['experiment_id'],bank_path=str(bank),bank_sha256=fitted['bank_sha256'],
        source_selection_locked_at=fitted['selection_locked_at'],locked_at=utc_now(),heads={})
    references=[config.RESULTS/'v2/cells'/f'{name}.json',config.ARTIFACTS/'v2/traces'/f'{name}.npz']
    lock['baseline_files']={str(p):file_sha256(p) for p in references if p.exists()}
    # Lock parameters and quadrature using the finite count space, before test data.
    for v in study['variants']:
        for m in study['training_methods']:
            key=f'{v}__{m}';f=fitted['fits'][fitted['selected'][key]]
            nodes=f['best']['normal_nodes'];bound=fitted['study']['integration']['sigma_bound']
            ladder=fitted['study']['integration']['normal_nodes'];errors=[]
            for ni in range(ladder.index(nodes),len(ladder)-1):
                nodes=ladder[ni];coarse=CountPosterior(f['best']['params'],v,nodes,bound)
                fine=CountPosterior(f['best']['params'],v,ladder[ni+1],bound)
                error=float(np.nanmax(np.abs(coarse.evidence-fine.evidence)))
                errors.append(dict(nodes=nodes,fine_nodes=ladder[ni+1],max_log_evidence_error=error))
                if error<=study['integration_tolerance']:break
            else:raise RuntimeError('P1 stopping count-space integration did not converge')
            lock['heads'][key]=dict(variant=v,training=m,params=f['best']['params'],normal_nodes=nodes,
                sigma_bound=bound,objective=f['objective'],source_selected=fitted['selected'][key],integration_checks=errors)
    write_json(root/'locks'/f'{name}.json',lock)
    observed_at=utc_now();values={};qa={}
    with np.load(bank,allow_pickle=False) as z:
        for split in ('dev','test'):
            a={k:z[f'{split}.{k}'] for k in ('ids','valid','question_id')}
            for k,val in a.items():values[f'{split}.{k}']=val
            for head,f in lock['heads'].items():
                tr,error=trace(a,f['params'],f['variant'],V2['voi'],f['normal_nodes'],f['sigma_bound'],key=f"{study['seed_key']}/{split}")
                for k,val in tr.items():values[f'{split}.{head}.{k}']=val
                qa[f'{split}.{head}']=error
                print(f'trace {name}/{split}/{head}: complete',flush=True)
    cache.parent.mkdir(parents=True,exist_ok=True)
    tmp=cache.with_name(f'{cache.name}.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**values)
    tmp.replace(cache)
    result={**lock,'evaluation_observations_loaded_at':observed_at,'trace_path':str(cache),
        'trace_sha256':file_sha256(cache),'count_lookup_posterior_errors':qa,'seconds':time.perf_counter()-started}
    write_json(path,result)
    print(f'trace {name}: saved ({result["seconds"]:.1f}s)',flush=True)
    return str(path)


def _job(args):return cell(*args)


def run(models=None,datasets=None,workers=4,overwrite=False,study=None):
    study=copy.deepcopy(study or CONFIG);root=out_dir();root.mkdir(parents=True,exist_ok=True)
    protocol=settings(study);protocol['experiment_id']=digest(protocol)[:16]
    target=root/'protocol.json'
    if target.exists() and read_json(target)!=protocol and not overwrite:raise ValueError('Stopping protocol differs; use a separate output or explicit overwrite')
    write_json(target,protocol)
    jobs=[(m,d,study,protocol['experiment_id'],overwrite) for m in models or V2['models'] for d in datasets or V2['datasets']]
    if workers==1:paths=list(map(_job,jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:paths=list(pool.map(_job,jobs))
    write_json(root/'run.json',dict(experiment_id=protocol['experiment_id'],cells=paths,completed_at=utc_now()))
    return paths
