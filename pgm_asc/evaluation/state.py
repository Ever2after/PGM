"""Locked state/value holdout evaluation and diagnostic reporting."""
from __future__ import annotations

import numpy as np

from pgm_asc.core.integrity import verify_lock
from pgm_asc.core.io import read_json, utc_now, write_json, write_jsonl
from pgm_asc.data.state import targets
from pgm_asc.evaluation.metrics import paired, policy_contrast, state_metrics
from pgm_asc.evaluation.replay import decision_metrics, history_metrics, run_state_policy
from pgm_asc.experiments.state_context import Context
from pgm_asc.policies.baselines import cges_logp, traces
from pgm_asc.policies.calibration import calibrate_state as calibrate
from pgm_asc.policies.stopping import distribution_history


def evaluate(config,ctx=None):
    ctx=ctx or Context(config);root=ctx.out
    lock=read_json(root/'selection_lock.json');verify_lock(lock)
    data,meta=ctx.data('test');n=len(meta)//8
    result={'created_at':utc_now(),'questions':n,'primary':'S_masked_seed42','smoke':ctx.smoke,
            'policy':{},'state':{},'selectors':{},'policy_families':{},'online_latency_verified':False}
    state_arrays={}
    for model,item in lock['models'].items():
        recipe=item['recipe'];obs=ctx.observed(recipe['selector'],'test');y,_=targets(data,obs)
        raw=ctx.state(recipe,'test');cal=calibrate(raw,item['temperature']['T'])
        result['state'][model]={'raw':state_metrics(raw,y),'calibrated':state_metrics(cal,y)}
        state_arrays[model]=-np.take_along_axis(raw,y[:,:,None],-1).squeeze(-1)[:,[1,3,7]].mean(1)
        result['policy'][model]=run_state_policy(ctx,recipe,'test',item['selected']['policy'],item['temperature']['T'])
        result['policy'][model]['calibration_feasible']=item['selected']['feasible']
        result['policy_families'][model]={f:run_state_policy(ctx,recipe,'test',v['policy'],item['temperature']['T'])
                                         for f,v in item['by_family'].items()}
    for s,item in lock['selectors'].items():
        lp=ctx.raw('test')[s];gold=data['gold_groups'].numpy().reshape(n,8,33)
        target=data['target'].numpy().reshape(n,8).copy();target[target==32]=8
        chosen=lp[:,:,:8].argmax(-1);valid=data['candidate_mask'][:,:8].any(1).numpy().reshape(n,8)
        correct=np.take_along_axis(gold,chosen[:,:,None],-1).squeeze(-1)&valid
        result['selectors'][s]={'raw_NLL':float(-np.take_along_axis(lp,target[:,:,None],-1)[:,[1,3,7]].mean()),
                               'accuracy':{str(t):float(correct[:,t-1].mean()) for t in [2,4,8]}}
        history=distribution_history(calibrate(lp,item['temperature']['T']),valid)
        result['policy']['selector_'+s]=history_metrics(history,data,meta,item['selected']['gamma'])
    for method,item in lock['baselines'].items():
        history=traces(ctx.banks('test'),method,item['settings'])
        if item['temperature']:
            lp=cges_logp(history,n=8).reshape(n,8,9);history=distribution_history(calibrate(lp,item['temperature']['T']),data['candidate_mask'][:,:8].any(1).numpy().reshape(n,8))
        result['policy'][method]=history_metrics(history,data,meta,item['selected']['gamma'])
    result['policy']['frozen_v2']=ctx.frozen('test')
    leaders=np.array([m['leader'] for m in meta]).reshape(n,8)
    for t in [2,4,8]:
        result['policy']['sc'+str(t)]=decision_metrics(np.full(n,t-1),np.full(n,'fixed'),np.where(leaders<8,leaders,-1),data,meta)
    result['state_contrasts']={control:paired(state_arrays['S_masked_seed42'],state_arrays[control],.975)
                              for control in ['S_mlp_seed42','S_unmasked_seed42']}
    result['paired_policy']={control:policy_contrast(result['policy'][result['primary']],result['policy'][control])
                             for control in ['frozen_v2','S_mlp_seed42','S_unmasked_seed42','M_masked_seed42','cges','sc8']}
    a,b=ctx.raw('test')['M'],ctx.raw('test')['S']
    target=data['target'].numpy().reshape(n,8).copy();target[target==32]=8
    anll=-np.take_along_axis(a,target[:,:,None],-1).squeeze(-1)[:,[1,3,7]].mean(1)
    bnll=-np.take_along_axis(b,target[:,:,None],-1).squeeze(-1)[:,[1,3,7]].mean(1)
    result['same_head_mixture_minus_single_NLL']=paired(anll,bnll)
    result['same_head_winner_agreement']={str(t):float((a[:,t-1,:8].argmax(-1)==b[:,t-1,:8].argmax(-1)).mean()) for t in [2,4,8]}
    result['failure_diagnostics']={}
    gold_present=data['target'].numpy().reshape(n,8)!=32
    for model,m in result['policy'].items():
        decisions=m['rows'];wrong=[(i,r) for i,r in enumerate(decisions) if not r['correct']]
        result['failure_diagnostics'][model]={'wrong':len(wrong),
            'wrong_gold_present':int(sum(gold_present[i,r['t']-1] for i,r in wrong)),
            'wrong_gold_absent':int(sum(not gold_present[i,r['t']-1] for i,r in wrong)),
            'reached_cap':sum(r['t']==8 for r in decisions),
            'stop_reasons':{reason:{'questions':sum(r['reason']==reason for r in decisions),
                                   'wrong':sum(r['reason']==reason and not r['correct'] for r in decisions)}
                            for reason in sorted({r['reason'] for r in decisions})}}
    for model,m in result['policy'].items():
        write_jsonl(root/'decisions'/f'{model}.jsonl',m['rows'])
        m.pop('rows')
    for models in result['policy_families'].values():
        for m in models.values():m.pop('rows')
    write_json(root/'evaluation.json',result)
    report(config,result,root)
    return result


def report(config,result,root):
    primary=result['primary'];p=result['policy'][primary];v2=result['policy']['frozen_v2']
    cohort_label='OLD-BANK INTEGRATION SMOKE' if result['smoke'] else 'Fresh holdout'
    lines=['# PGM v5 결과 요약','',f"{cohort_label}: {result['questions']} questions, 8 responses/question. Primary: `{primary}`.",'',
           '모든 모델과 policy는 holdout generation/evaluation 전에 고정했다. Generated-token replay이며 online latency는 측정하지 않았다.','',
           '| Method | Accuracy | Mean samples | Mean generated tokens |','|---|---:|---:|---:|']
    for key in [primary,'S_mlp_seed42','S_unmasked_seed42','M_masked_seed42','selector_S','selector_M','frozen_v2','cges','asc','reasc','sc8']:
        m=result['policy'][key];lines.append(f"| {key} | {m['accuracy']:.3f} | {m['mean_samples']:.3f} | {m['mean_tokens']:.1f} |")
    lines+=['','## State mechanism','',str(result['state_contrasts']),'',
            'Intervals are 97.5% for the two primary state-NLL contrasts; negative favors masked temporal PGM.','',
            '## Same-head selector ablation','',str(result['same_head_mixture_minus_single_NLL']),
            'Winner agreement: '+str(result['same_head_winner_agreement']),'',
            '## Primary paired policy comparisons','',str(result['paired_policy']),'',
            '## Interpretation','',
            f"Primary−frozen-v2 accuracy: {(p['accuracy']-v2['accuracy'])*100:+.2f}%p; generated-token ratio: {p['mean_tokens']/v2['mean_tokens']:.3f}.",
            'Point accuracy/cost constraints are not statistical non-inferiority. Same-question seed repeats are not pooled as independent data.',
            'Fixed-4-step and Bellman comparisons use calibration-selected policies from each predeclared family, not test-selected thresholds.',
            'State/policy train900 excludes the frozen selector train800. The spent v2 test500 is part of dev700, not a v5 holdout.',
            'Masking benefits must exceed the same-feature MLP and same-parameter unmasked chain before claiming temporal-structure value.',
            'Backward value training uses realized payoff under a fitted policy, not a per-question future oracle maximum. Inner fitted-state/value optimism remains a limitation; complete outer-fold policy audits are saved.',
            'No constraint verifier, new prompt, or new reasoning embedding was introduced.','',
            '[Full metrics](./evaluation.json) · [Policy lock](./selection_lock.json) · [Model lock](./model_lock.json)']
    root.mkdir(parents=True,exist_ok=True)
    (root/'result_summary.md').write_text('\n'.join(lines)+'\n')
    (root/'report.md').write_text('\n'.join(lines)+'\n')
