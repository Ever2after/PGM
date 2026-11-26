"""P1 objective study, with immutable previous Joint reference and dev locks."""
from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from pgm_asc import config
from pgm_asc.diagnostics import metrics
from pgm_asc.hierarchical.model import KEYS,forecast,infer,readable,tensors
from pgm_asc.objectives.run import average_forecasts,future_calibration,load_splits,objectives
from pgm_asc.objectives.train import common_training_subset,prefix_weights
from pgm_asc.p1_objectives import CONFIG
from pgm_asc.p1_objectives.train import Counts,dev_score,fit_one,integration_check
from pgm_asc.utils import digest,file_sha256,utc_now,write_json


def code_hashes():
    files=['p1_objectives/train.py','p1_objectives/run.py','hierarchical/model.py',
           'objectives/train.py','objectives/run.py','diagnostics/metrics.py',
           'diagnostics/predictive.py','v2/model.py']
    return {p:file_sha256(config.ROOT/'pgm_asc'/p) for p in files}


def identity(study):
    return digest(dict(study=study,code=code_hashes()))[:16]


def method_key(variant,method):return f'{variant}__{method}'


def baseline_root(study):
    return Path(study.get('baseline_dir',config.RESULTS/'hierarchical_crp'))


def load_baseline_fit(model,dataset,study):
    folder=baseline_root(study)
    path=folder/'selection'/f'{model}_{dataset}.json'
    source=json.loads(path.read_text())
    if source['study']['evaluation']!=study['evaluation']:
        raise ValueError('Previous Joint evaluation settings differ from this paired study')
    for k in ('unique_correct_only','ridge','restart_seeds','initialization_sd','sigma_starts'):
        if source['study']['training'][k]!=study['training'][k]:
            raise ValueError(f'Previous Joint training control differs: {k}')
    if source['study']['integration']['sigma_bound']!=study['integration']['sigma_bound']:
        raise ValueError('P1 sigma bound must remain unchanged')
    protocol=json.loads((folder/'protocol.json').read_text())
    for rel,sha in protocol['code_sha256'].items():
        if file_sha256(config.ROOT/rel)!=sha:
            raise ValueError(f'Previous Joint numerical implementation changed: {rel}')
    return source,file_sha256(path)


def fit_objective(model,dataset,variant,objective,train_counts,dev_counts,dev,study,source,key,source_sha,overwrite):
    ladder=study['integration']['normal_nodes']
    start=ladder.index(source['normal_nodes'])
    bound=study['integration']['sigma_bound']
    history=[]
    for index in range(start,len(ladder)-1):
        nodes=ladder[index]
        path=config.ARTIFACTS/'p1_objectives'/key/'candidates'/f'{model}_{dataset}'/f'{variant}_{objective.name}_{nodes}.json'
        if path.exists() and not overwrite:
            record=json.loads(path.read_text())
            if record['source_fit_sha256']!=source_sha:
                raise ValueError('Candidate cache has a different Joint source')
        else:
            restarts=[]
            for seed in study['training']['restart_seeds']:
                r=fit_one(train_counts,variant,objective,study['training'],seed,nodes,bound)
                r['dev_unique_nll']=dev_score(dev_counts,r['params'],variant,nodes,study['evaluation']['prefixes'],bound)
                restarts.append(r)
            best=min(restarts,key=lambda r:(r['penalized_loss'],r['restart_seed']))
            check=integration_check(best['params'],variant,objective,train_counts,dev_counts,dev,
                nodes,ladder[index+1],study['evaluation']['prefixes'],bound)
            record=dict(objective=vars(objective),variant=variant,best=best,restarts=restarts,
                        quadrature=check,source_fit_sha256=source_sha)
            write_json(path,record)
        history.append(record)
        print(f"fit {model}/{dataset}/{variant}/{objective.name} u={nodes}: dev={record['best']['dev_unique_nll']:.5f}, integration={record['quadrature']['max_error']:.3g}",flush=True)
        if record['quadrature']['max_error']<=study['integration']['tolerance']:
            return {**record,'resolution_history':history}
    raise RuntimeError(f'P1 objective integration failed before test: {model}/{dataset}/{variant}/{objective.name}')


def fit_cell(model,dataset,study,out,overwrite=False):
    """Only train and dev arrays and the prior Joint fit lock are read here."""
    torch.set_num_threads(1)
    data,bank_sha=load_splits(model,dataset,('train','dev'))
    source,source_sha=load_baseline_fit(model,dataset,study)
    key=identity(study)
    path=config.ARTIFACTS/'p1_objectives'/key/'fits'/f'{model}_{dataset}.json'
    if path.exists() and not overwrite:
        cached=json.loads(path.read_text())
        if cached['bank_sha256']!=bank_sha or cached['source_fit_sha256']!=source_sha:
            raise ValueError('Fit cache provenance differs')
        write_json(Path(out)/'selection'/path.name,cached)
        return cached
    train,keep=common_training_subset(data['train'])
    dev,dev_keep=common_training_subset(data['dev'])
    cohort=digest(train['question_id'].tolist())
    if source['bank_sha256']!=bank_sha or source['train_question_hash']!=cohort:
        raise ValueError('P1 objective study must use the exact previous Joint bank/cohort')
    counts,dev_counts=Counts(train),Counts(dev)
    fits,selected,selection_checks={},{},[]
    for variant in study['variants']:
        previous=copy.deepcopy(source['fits'][variant])
        previous.update(objective=dict(kind='joint',weighting='uniform',mixing=None),variant=variant,
                        frozen_previous_joint=True,resolution_history=[])
        previous['best']['normal_nodes']=source['normal_nodes']
        previous['best']['dev_unique_nll']=dev_score(dev_counts,previous['best']['params'],variant,
            source['normal_nodes'],study['evaluation']['prefixes'],study['integration']['sigma_bound'])
        fits[method_key(variant,'joint')]=previous
        selected[method_key(variant,'joint')]=method_key(variant,'joint')
        for obj in objectives(study):
            if obj.kind=='joint':continue
            name=method_key(variant,obj.name)
            fits[name]=fit_objective(model,dataset,variant,obj,counts,dev_counts,data['dev'],study,
                source,key,source_sha,overwrite)
        for weighting in study['training']['weightings']:
            name=method_key(variant,f'prefix_{weighting}')
            selected[name]=name
            candidates=[method_key(variant,o.name) for o in objectives(study) if o.kind=='mixed' and o.weighting==weighting]
            best=min(candidates,key=lambda n:(fits[n]['best']['dev_unique_nll'],fits[n]['objective']['mixing']))
            fine_best=min(candidates,key=lambda n:(fits[n]['quadrature']['fine']['dev_unique_nll'],fits[n]['objective']['mixing']))
            selection_checks.append(dict(variant=variant,weighting=weighting,coarse_choice=best,
                finer_choice=fine_best,choice_changed=best!=fine_best,
                dev_margin=float(sorted(fits[n]['best']['dev_unique_nll'] for n in candidates)[1]-fits[best]['best']['dev_unique_nll'])))
            if best!=fine_best:
                raise RuntimeError('Mixed lambda ranking changes under finer integration; refine train/dev before test')
            selected[method_key(variant,f'mixed_{weighting}')]=best
    result=dict(experiment_id=key,model=model,dataset=dataset,study=study,bank_sha256=bank_sha,
        train_questions=len(keep),train_kept=int(keep.sum()),dev_questions=len(dev_keep),dev_unique=int(dev_keep.sum()),
        train_question_hash=cohort,train_question_ids=train['question_id'].tolist(),source_fit_sha256=source_sha,
        source_experiment_id=source['experiment_id'],fits=fits,selected=selected,selection_checks=selection_checks,
        selection='Frozen previous Joint; restart: penalized train loss; mixed lambda: dev unique NLL at fixed evaluation prefixes',
        selection_locked_at=utc_now())
    write_json(path,result)
    write_json(Path(out)/'selection'/path.name,result)
    return result


def evaluate_cell(model,dataset,study,out,fitted,overwrite=False):
    torch.set_num_threads(1)
    out,key=Path(out),identity(study)
    path=out/'cells'/f'{model}_{dataset}.json'
    if path.exists() and not overwrite:
        old=json.loads(path.read_text())
        if old['experiment_id']==key and old['fit_digest']==digest(fitted):return str(path)
        raise ValueError('Existing evaluation belongs to a different fit/configuration')
    data,bank_sha=load_splits(model,dataset,study['evaluation']['splits'])
    if bank_sha!=fitted['bank_sha256']:raise ValueError('Bank changed after fitting')
    prior_path=baseline_root(study)/'cells'/path.name
    prior=json.loads(prior_path.read_text())
    if prior['bank_sha256']!=bank_sha or prior['experiment_id']!=fitted['source_experiment_id']:
        raise ValueError('Joint prediction bank/experiment differs')
    if file_sha256(prior['prediction_cache']['path'])!=prior['prediction_cache']['sha256']:
        raise ValueError('Previous Joint prediction cache changed')
    e=study['evaluation'];bound=study['integration']['sigma_bound']
    methods={name:fitted['fits'][source] for name,source in fitted['selected'].items()}
    result=dict(experiment_id=key,model=model,dataset=dataset,study=study,bank_sha256=bank_sha,
        fit_digest=digest(fitted),source_cell_sha256=file_sha256(prior_path),selection=fitted['selected'],
        train_kept=fitted['train_kept'],selection_locked_at=fitted['selection_locked_at'],
        evaluation_started_at=utc_now(),parameters={name:dict(raw=f['best']['params'],
            readable=readable(f['best']['params'],bound),count=len(f['best']['params']),variant=f['variant'],
            objective=f['objective'],normal_nodes=f['best']['normal_nodes']) for name,f in methods.items()},splits={})
    cache_values={}
    with np.load(prior['prediction_cache']['path'],allow_pickle=False) as previous:
        for split,arrays in data.items():
            if not np.array_equal(previous[f'{split}.question_id'],arrays['question_id']):
                raise ValueError('Joint prediction question order differs')
            entries,losses_by_model={},{}
            unique=metrics.targets(arrays)[3]
            obs={k:v for k,v in arrays.items() if k!='correct'}
            for name,f in methods.items():
                variant=f['variant']
                if f['objective']['kind']=='joint':
                    post=previous[f'{split}.{variant}.post']
                    prefix=f'{split}.{variant}.future.'
                    future={k[len(prefix):]:previous[k] for k in previous.files if k.startswith(prefix)}
                    sensitivity=copy.deepcopy(prior['splits'][split]['models'][variant]['mc_seed_sensitivity'])
                else:
                    p=infer(tensors(f['best']['params']),variant,obs,f['best']['normal_nodes'],bound)
                    post=p.post
                    draws,sensitivity=[],[]
                    for seed in e['mc_seeds']:
                        pred=forecast(obs,p,e['prefixes'],e['horizons'],e['simulations_per_seed'],seed,f'{model}/{dataset}/{split}')
                        _,_,fl=metrics.future_scores(pred,arrays,e['prefixes'],e['horizons'])
                        sensitivity.append(dict(seed=seed,**{k:float(v.mean()) for k,v in fl.items()}))
                        draws.append(pred)
                    future=average_forecasts(draws)
                    del p
                if not np.allclose(future['gain'],future['repair']-future['damage'],equal_nan=True):
                    raise ValueError('Paired gain identity failed')
                pr,pb,pl=metrics.posterior_losses(post,arrays,e['prefixes'])
                fr,gb,fl=metrics.future_scores(future,arrays,e['prefixes'],e['horizons'])
                loss={**pl,**fl}
                entries[name]=dict(summary={k:metrics.bootstrap_mean(v,e['bootstrap_replicates'],e['bootstrap_seed']) for k,v in loss.items()},
                    future_unique_summary={k:metrics.bootstrap_mean(v[unique],e['bootstrap_replicates'],e['bootstrap_seed']) for k,v in fl.items()},
                    posterior_by_prefix=pr,posterior_calibration=pb,future_by_horizon=fr,gain_calibration=gb,
                    transition_calibration=future_calibration(future,arrays,e['prefixes'],e['horizons']),mc_seed_sensitivity=sensitivity)
                losses_by_model[name]=loss
                cache_values[f'{split}.{name}.post']=post
                for k,v in future.items():cache_values[f'{split}.{name}.future.{k}']=v
                print(f'evaluate {model}/{dataset}/{split}/{name}: paired MC complete',flush=True)
            comparisons={}
            def compare(left,right):
                comparisons[f'{left} minus {right}']={k:metrics.bootstrap_mean(v-losses_by_model[right][k],
                    e['bootstrap_replicates'],e['bootstrap_seed']) for k,v in losses_by_model[left].items()}
            for name,f in methods.items():
                base=method_key(f['variant'],'joint')
                if name!=base:compare(name,base)
            for method in ('joint',)+tuple(f'{k}_{w}' for w in study['training']['weightings'] for k in ('prefix','mixed')):
                left,right=method_key('P1_linked',method),method_key('P1_independent',method)
                if left in methods and right in methods:compare(left,right)
            for variant in study['variants']:
                for w in study['training']['weightings']:
                    compare(method_key(variant,f'mixed_{w}'),method_key(variant,f'prefix_{w}'))
            result['splits'][split]=dict(questions=len(unique),unique_questions=int(unique.sum()),models=entries,paired_contrasts=comparisons)
            cache_values[f'{split}.question_id']=arrays['question_id']
    cache=config.ARTIFACTS/'p1_objectives'/key/'predictions'/path.with_suffix('.npz').name
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_name(f'{cache.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**cache_values)
    temporary.replace(cache)
    result['prediction_cache']=dict(path=str(cache),sha256=file_sha256(cache))
    write_json(path,result)
    return str(path)


def _job(job):
    model,dataset,study,out,stage,overwrite=job
    fitted=fit_cell(model,dataset,study,out,overwrite)
    return (model,dataset) if stage=='fit' else evaluate_cell(model,dataset,study,out,fitted,overwrite)


def run(models=None,datasets=None,workers=2,out=None,stage='run',overwrite=False,study=None):
    study=copy.deepcopy(study or CONFIG)
    out=Path(out or config.RESULTS/'p1_objectives').resolve()
    if stage=='report':
        from pgm_asc.p1_objectives.report import report
        return report(out)
    if set(study['variants'])!={'P1_independent','P1_linked'}:
        raise ValueError('This study requires both fixed P1 variants')
    if not study['training']['unique_correct_only']:raise ValueError('Require common unique-correct cohort')
    protocol=dict(experiment_id=identity(study),study=study,code_sha256=code_hashes(),
        parameters={k:list(KEYS[k]) for k in study['variants']},inputs=['answer ids','valid'],
        prefix_weights={w:prefix_weights(w).tolist() for w in study['training']['weightings']},
        baseline='Immutable prior Joint parameters and prediction arrays; exact bank/cohort/evaluation checks',
        numerics='Each candidate checked on train/dev against finer Normal quadrature; refine before test; lambda ranking must remain stable')
    path=out/'protocol.json'
    if path.exists() and not overwrite and json.loads(path.read_text())!=protocol:
        raise ValueError('Protocol/code differs; choose a new output or --overwrite')
    write_json(path,protocol)
    models,datasets=models or study['models'],datasets or study['datasets']
    jobs=[(m,d,study,str(out),stage,overwrite) for m in models for d in datasets]
    if workers==1:paths=[_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:paths=list(pool.map(_job,jobs))
    if stage!='fit':
        write_json(out/'run.json',dict(cells=[str(Path(p).relative_to(out)) for p in paths],
            models=models,datasets=datasets,experiment_id=identity(study)))
        from pgm_asc.p1_objectives.report import report
        report(out)
    return paths
