"""P1 Mixed + exact multi-step repair/damage/stay log score."""
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
from pgm_asc.objectives.run import average_forecasts,future_calibration,load_splits
from pgm_asc.objectives.train import common_training_subset,prefix_weights
from pgm_asc.p1_transitions import CONFIG
from pgm_asc.p1_transitions.train import Counts,dev_score,fit_one,integration_check
from pgm_asc.p1_transitions.evaluation import transition_nll,exact_forecast,exact_score_keys
from pgm_asc.utils import digest,file_sha256,utc_now,write_json


def code_hashes():
    files=['p1_transitions/train.py','p1_transitions/run.py','p1_transitions/kernel.py','p1_transitions/evaluation.py',
        'p1_predictive/train.py','p1_predictive/run.py','hierarchical/model.py',
        'p1_objectives/train.py','p1_objectives/run.py','objectives/train.py','objectives/run.py',
        'diagnostics/metrics.py','diagnostics/predictive.py','v2/model.py']
    return {p:file_sha256(config.ROOT/'pgm_asc'/p) for p in files}


def identity(study):return digest(dict(study=study,code=code_hashes()))[:16]
def method_key(variant,method):return f'{variant}__{method}'
def transition_name(weighting,beta):return f"transition_{weighting}_beta_{beta:g}".replace('.','p')
def baseline_root(study):return Path(study.get('baseline_dir',config.RESULTS/'p1_predictive'))


def load_source(model,dataset,study):
    folder=baseline_root(study);path=folder/'selection'/f'{model}_{dataset}.json'
    source=json.loads(path.read_text())
    if source['study']['evaluation']!=study['evaluation']:raise ValueError('Frozen baseline evaluation settings differ')
    for k in ('unique_correct_only','ridge','restart_seeds','initialization_sd','sigma_starts','weightings'):
        if source['study']['training'][k]!=study['training'][k]:raise ValueError(f'Frozen training control differs: {k}')
    if source['study']['integration']['sigma_bound']!=study['integration']['sigma_bound']:
        raise ValueError('P1 sigma bound changed')
    protocol=json.loads((folder/'protocol.json').read_text())
    for rel,sha in protocol['code_sha256'].items():
        if file_sha256(config.ROOT/'pgm_asc'/rel)!=sha:raise ValueError(f'Frozen baseline implementation changed: {rel}')
    for variant in study['variants']:
        for w in study['training']['weightings']:
            if method_key(variant,f'predictive_{w}_gamma_1') not in source['selected']:
                raise ValueError('Frozen predictive gamma=1 comparator is required')
    return source,file_sha256(path)


def fit_candidate(model,dataset,variant,weighting,beta,mixed,train,dev,dev_full,study,key,source_sha,overwrite):
    ladder=study['integration']['normal_nodes'];bound=study['integration']['sigma_bound']
    start=ladder.index(mixed['best']['normal_nodes']);mixing=mixed['objective']['mixing']
    history=[]
    for ni in range(start,len(ladder)-1):
        nodes=ladder[ni]
        path=config.ARTIFACTS/'p1_transitions'/key/'candidates'/f'{model}_{dataset}'/f'{variant}_{transition_name(weighting,beta)}_{nodes}.json'
        if path.exists() and not overwrite:
            record=json.loads(path.read_text())
            if record['source_fit_sha256']!=source_sha:raise ValueError('Candidate cache source differs')
        else:
            restarts=[]
            for seed in study['training']['restart_seeds']:
                r=fit_one(train,variant,study['training'],seed,mixing,beta,weighting,nodes,bound)
                r.update(dev_score(dev,r['params'],variant,nodes,study['evaluation']['prefixes'],bound))
                restarts.append(r)
            best=min(restarts,key=lambda r:(r['penalized_loss'],r['restart_seed']))
            check=integration_check(best['params'],variant,train,dev,dev_full,nodes,ladder[ni+1],
                study['evaluation']['prefixes'],weighting,mixing,beta,bound)
            record=dict(variant=variant,objective=dict(kind='transition',weighting=weighting,mixing=mixing,beta=beta),
                best=best,restarts=restarts,quadrature=check,source_fit_sha256=source_sha)
            write_json(path,record)
        history.append(record)
        print(f"fit {model}/{dataset}/{variant}/{transition_name(weighting,beta)} u={nodes}: dev post={record['best']['dev_unique_nll']:.5f}, transition={record['best']['dev_transition_nll']:.5f}, integration={record['quadrature']['max_error']:.3g}",flush=True)
        if record['quadrature']['max_error']<=study['integration']['tolerance']:
            return {**record,'resolution_history':history}
    raise RuntimeError('Predictive-objective quadrature did not converge before test')


def fit_cell(model,dataset,study,out,overwrite=False):
    """Read train/dev observations and existing fit locks, without test scores."""
    torch.set_num_threads(1)
    data,bank_sha=load_splits(model,dataset,('train','dev'))
    source,source_sha=load_source(model,dataset,study)
    key=identity(study);path=config.ARTIFACTS/'p1_transitions'/key/'fits'/f'{model}_{dataset}.json'
    if path.exists() and not overwrite:
        cached=json.loads(path.read_text())
        if cached['bank_sha256']!=bank_sha or cached['source_fit_sha256']!=source_sha:raise ValueError('Fit cache provenance differs')
        write_json(Path(out)/'selection'/path.name,cached);return cached
    train,keep=common_training_subset(data['train']);dev,dev_keep=common_training_subset(data['dev'])
    cohort=digest(train['question_id'].tolist())
    if source['bank_sha256']!=bank_sha or source['train_question_hash']!=cohort:raise ValueError('Previous bank/cohort differs')
    train_args=(study['training']['transition_prefixes'],study['training']['transition_horizons'])
    counts,dev_counts=Counts(train,*train_args),Counts(dev,*train_args)
    fits={}
    for name,src in source['selected'].items():
        if not name.split('__',1)[1].startswith(('mixed_','predictive_')):continue
        f=copy.deepcopy(source['fits'][src]);f['frozen_previous']=True
        fits[name]=f
    for variant in study['variants']:
        for weighting in study['training']['weightings']:
            mixed=fits[method_key(variant,f'mixed_{weighting}')]
            for beta in study['training']['transition_weights']:
                name=method_key(variant,transition_name(weighting,beta))
                fits[name]=fit_candidate(model,dataset,variant,weighting,beta,mixed,counts,dev_counts,
                    data['dev'],study,key,source_sha,overwrite)
    result=dict(experiment_id=key,model=model,dataset=dataset,study=study,bank_sha256=bank_sha,
        train_questions=len(keep),train_kept=int(keep.sum()),dev_questions=len(dev_keep),dev_unique=int(dev_keep.sum()),
        train_question_hash=cohort,train_question_ids=train['question_id'].tolist(),source_fit_sha256=source_sha,
        source_experiment_id=source['experiment_id'],fits=fits,selected={n:n for n in fits},
        selection='Frozen Mixed/all next-answer arms; fixed lambda; all beta arms; restart by penalized train loss',
        selection_locked_at=utc_now())
    write_json(path,result);write_json(Path(out)/'selection'/path.name,result)
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
        raise ValueError('Frozen prediction bank/experiment differs')
    if file_sha256(prior['prediction_cache']['path'])!=prior['prediction_cache']['sha256']:
        raise ValueError('Frozen prediction cache changed')
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
                raise ValueError('Frozen prediction question order differs')
            entries,losses_by_model={},{}
            unique=metrics.targets(arrays)[3]
            obs={k:v for k,v in arrays.items() if k!='correct'}
            exact_counts=Counts(obs,study['training']['transition_prefixes'],study['training']['transition_horizons'])
            exact_losses={}
            for name,f in methods.items():
                variant=f['variant']
                if f.get('frozen_previous'):
                    post=previous[f'{split}.{name}.post']
                    prefix=f'{split}.{name}.future.'
                    future={k[len(prefix):]:previous[k] for k in previous.files if k.startswith(prefix)}
                    sensitivity=copy.deepcopy(prior['splits'][split]['models'][name]['mc_seed_sensitivity'])
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
                fl['transition_categorical_nll']=transition_nll(future,arrays,e['prefixes'],e['horizons'])
                loss={**pl,**fl}
                exact=exact_forecast(exact_counts,tensors(f['best']['params']),variant,f['best']['normal_nodes'],bound,e['prefixes'],e['horizons'],future)
                _,_,el=metrics.future_scores(exact,arrays,e['prefixes'],e['horizons'])
                el=exact_score_keys(el)
                el['transition_categorical_nll']=transition_nll(exact,arrays,e['prefixes'],e['horizons'])
                exact_losses[name]=el
                entries[name]=dict(summary={k:metrics.bootstrap_mean(v,e['bootstrap_replicates'],e['bootstrap_seed']) for k,v in loss.items()},
                    future_unique_summary={k:metrics.bootstrap_mean(v[unique],e['bootstrap_replicates'],e['bootstrap_seed']) for k,v in fl.items()},
                    posterior_by_prefix=pr,posterior_calibration=pb,future_by_horizon=fr,gain_calibration=gb,
                    transition_calibration=future_calibration(future,arrays,e['prefixes'],e['horizons']),mc_seed_sensitivity=sensitivity)
                entries[name]['exact_transition_summary']={k:metrics.bootstrap_mean(v,e['bootstrap_replicates'],e['bootstrap_seed']) for k,v in el.items()}
                entries[name]['exact_transition_calibration']=future_calibration(exact,arrays,e['prefixes'],e['horizons'])
                losses_by_model[name]=loss
                for k in ('repair','damage','gain','current_correct_mc','future_reference_correct'):
                    cache_values[f'{split}.{name}.exact.{k}']=exact[k]
                cache_values[f'{split}.{name}.post']=post
                for k,v in future.items():cache_values[f'{split}.{name}.future.{k}']=v
                print(f'evaluate {model}/{dataset}/{split}/{name}: paired MC complete',flush=True)
            comparisons={};exact_comparisons={}
            def compare(left,right):
                comparison=f'{left} minus {right}'
                comparisons[comparison]={k:metrics.bootstrap_mean(v-losses_by_model[right][k],
                    e['bootstrap_replicates'],e['bootstrap_seed']) for k,v in losses_by_model[left].items()}
                exact_comparisons[comparison]={k:metrics.bootstrap_mean(v-exact_losses[right][k],
                    e['bootstrap_replicates'],e['bootstrap_seed']) for k,v in exact_losses[left].items()}
            for name,f in methods.items():
                if f.get('frozen_previous'):continue
                variant=f['variant'];w=f['objective']['weighting']
                compare(name,method_key(variant,f'mixed_{w}'))
                # gamma=1 is a fixed, historical predictive-loss comparator.
                compare(name,method_key(variant,f'predictive_{w}_gamma_1'))
            for w in study['training']['weightings']:
                for beta in study['training']['transition_weights']:
                    method=transition_name(w,beta)
                    compare(method_key('P1_linked',method),method_key('P1_independent',method))
            result['splits'][split]=dict(questions=len(unique),unique_questions=int(unique.sum()),models=entries,
                paired_contrasts=comparisons,exact_paired_contrasts=exact_comparisons)
            cache_values[f'{split}.question_id']=arrays['question_id']
    cache=config.ARTIFACTS/'p1_transitions'/key/'predictions'/path.with_suffix('.npz').name
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
    study=copy.deepcopy(study or CONFIG);out=Path(out or config.RESULTS/'p1_transitions').resolve()
    if stage=='report':
        from pgm_asc.p1_transitions.report import report
        return report(out)
    if set(study['variants'])!={'P1_independent','P1_linked'}:raise ValueError('Require both fixed P1 graphs')
    if not study['training']['unique_correct_only']:raise ValueError('Require the same unique-correct training cohort')
    training_pairs={(t,h) for t in study['training']['transition_prefixes'] for h in study['training']['transition_horizons'] if t+h<=16}
    evaluation_pairs={(t,h) for t in study['evaluation']['prefixes'] for h in study['evaluation']['horizons'] if t+h<=16}
    if training_pairs!=evaluation_pairs:raise ValueError('Transition training and evaluation pairs must match')
    betas=study['training']['transition_weights']
    if not betas or len(set(betas))!=len(betas) or min(betas)<=0:raise ValueError('Require distinct positive fixed beta arms')
    protocol=dict(experiment_id=identity(study),study=study,code_sha256=code_hashes(),
        parameters={k:list(KEYS[k]) for k in study['variants']},inputs=['answer ids','valid'],
        prefix_weights={w:prefix_weights(w).tolist() for w in study['training']['weightings']},
        loss='M=(J+lambda*C)/(1+lambda); L=(M+beta*T)/(1+beta)+ridge; T=mean categorical NLL of repair/damage/stay',
        beta_selection='none; all prespecified beta arms evaluated',
        baseline='Frozen Mixed and all next-answer predictive weights; gamma=1 fixed for paired contrasts',
        transition='Native multi-step events by exact restricted CRP partitions and q/Normal quadrature; gold current/future labels are targets only')
    path=out/'protocol.json'
    if path.exists() and not overwrite and json.loads(path.read_text())!=protocol:raise ValueError('Protocol/code differs; choose new output or --overwrite')
    write_json(path,protocol)
    models,datasets=models or study['models'],datasets or study['datasets']
    jobs=[(m,d,study,str(out),stage,overwrite) for m in models for d in datasets]
    if workers==1:paths=[_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:paths=list(pool.map(_job,jobs))
    if stage!='fit':
        write_json(out/'run.json',dict(cells=[str(Path(p).relative_to(out)) for p in paths],models=models,datasets=datasets,experiment_id=identity(study)))
        from pgm_asc.p1_transitions.report import report
        report(out)
    return paths
