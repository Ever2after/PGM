"""State-stage temperature/policy calibration and locking."""
from __future__ import annotations

import numpy as np

from pgm_asc.core.config import STATE as ROOT
from pgm_asc.core.integrity import file_sha, verify_lock
from pgm_asc.core.io import digest, log, read_json, utc_now, write_json
from pgm_asc.data.state import targets
from pgm_asc.evaluation.replay import decision_metrics, history_metrics, run_state_policy
from pgm_asc.policies.stopping import GAMMAS
from pgm_asc.experiments.state_context import Context
from pgm_asc.policies.baselines import cges_logp, reasc_calibrate, traces
from pgm_asc.policies.calibration import (
    calibrate_state as calibrate,
    fit_state_temperature as fit_temperature,
)
from pgm_asc.policies.stopping import (
    choose_state_candidate as choose,
    distribution_history,
    policy_grid,
)
from pgm_asc.training.recipes import name, recipes


def lock_calibration(config,ctx=None):
    ctx=ctx or Context(config);root=ctx.out
    path=root/'selection_lock.json'
    if path.exists():
        value=read_json(path);verify_lock(value);return value
    da,ma=ctx.data('calibration_a');db,mb=ctx.data('calibration_b')
    leader=np.array([m['leader'] for m in mb]).reshape(-1,8)
    sc8=decision_metrics(np.full(len(leader),7),np.full(len(leader),'fixed'),np.where(leader<8,leader,-1),db,mb)['accuracy']
    frozen=ctx.frozen('calibration_b');reference=max(sc8,frozen['accuracy'])
    result={'created_at':utc_now(),'models':{},'selectors':{},'baselines':{},'calibration_B_reference':reference,
            'calibration_B_SC8':sc8,'calibration_B_frozen_v2':frozen['accuracy'],'holdout_accessed':False,'smoke':ctx.smoke}
    for recipe in recipes():
        obs=ctx.observed(recipe['selector'],'calibration_a');y,_=targets(da,obs)
        fitted=fit_temperature(ctx.state(recipe,'calibration_a'),y)
        candidates=[]
        for policy in policy_grid(config['lambda_grid'],GAMMAS):
            m=run_state_policy(ctx,recipe,'calibration_b',policy,fitted['T']);m.pop('rows')
            candidates.append({'policy':policy,**m})
        family_best={family:choose([c for c in candidates if c['policy']['family']==family],reference)
                     for family in ['threshold','fixed4','bellman']}
        result['models'][name(recipe)]={'recipe':recipe,'temperature':fitted,'selected':choose(candidates,reference),
                                       'by_family':family_best,'grid':candidates}
        log('v5 model policy calibrated',model=name(recipe),selected=result['models'][name(recipe)]['selected'])
        write_json(root/'calibration_progress.json',{'completed_models':len(result['models']),'planned_models':12})
    answer_target=da['target'].numpy().reshape(-1,8).copy();answer_target[answer_target==32]=8
    for selector in ['M','S','A','I','N']:
        fitted=fit_temperature(ctx.raw('calibration_a')[selector],answer_target)
        history=distribution_history(calibrate(ctx.raw('calibration_b')[selector],fitted['T']),db['candidate_mask'][:,:8].any(1).numpy().reshape(-1,8))
        grid=[]
        for gamma in GAMMAS:
            m=history_metrics(history,db,mb,gamma);m.pop('rows');grid.append({'gamma':gamma,**m})
        result['selectors'][selector]={'temperature':fitted,'selected':choose(grid,reference),'grid':grid}
    settings=read_json('configs/experiment.json')['baselines']['cges']
    for method in ['cges','asc','reasc']:
        param=settings if method=='cges' else (reasc_calibrate(ctx.banks('calibration_a'),da,8) if method=='reasc' else {})
        ah=traces(ctx.banks('calibration_a'),method,param);bh=traces(ctx.banks('calibration_b'),method,param)
        fitted=None
        if method=='cges':
            alp=cges_logp(ah,n=8).reshape(-1,8,9);blp=cges_logp(bh,n=8).reshape(-1,8,9)
            fitted=fit_temperature(alp,answer_target);bh=distribution_history(calibrate(blp,fitted['T']),db['candidate_mask'][:,:8].any(1).numpy().reshape(-1,8))
        grid=[]
        for gamma in GAMMAS:
            m=history_metrics(bh,db,mb,gamma);m.pop('rows');grid.append({'gamma':gamma,**m})
        result['baselines'][method]={'settings':param,'temperature':fitted,'selected':choose(grid,reference),'grid':grid}
    artifacts={}
    if not ctx.smoke:
        verify_lock(read_json(ROOT/'model_lock.json'))
        artifacts.update(read_json(ROOT/'model_lock.json')['artifact_sha256'])
        cohort=read_json(ROOT/'cohort_lock.json');artifacts.update(cohort['input_sha256'])
        artifacts[str(ROOT/'cohort_lock.json')]=file_sha(ROOT/'cohort_lock.json')
        artifacts[str(ROOT/'model_lock.json')]=file_sha(ROOT/'model_lock.json')
    result['artifact_sha256']=artifacts;result['protocol_hash']=digest(result)
    write_json(path,result);return result
