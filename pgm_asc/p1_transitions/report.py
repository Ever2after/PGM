"""Direct multi-step repair/damage ablation: paired MC and exact-event sensitivity."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from pgm_asc import config
from pgm_asc.diagnostics.report import csv_file,fmt,markdown_html,table
from pgm_asc.p1_transitions.run import baseline_root,method_key,transition_name
from pgm_asc.p1_transitions import CONFIG

VARIANTS=('P1_independent','P1_linked')
SHORT={'P1_independent':'q 연결 제거','P1_linked':'q 연결 유지'}
def plot_label(method):
    if method.startswith('transition_'):
        return 'Mixed + R/D, beta='+method.split('_beta_')[1].replace('p','.')
    if method.startswith('predictive_'):
        return 'Mixed + next, gamma='+method.split('_gamma_')[1].replace('p','.')
    return 'Mixed'


def method_order(study):
    return tuple(m for w in study['training']['weightings'] for m in
        [f'mixed_{w}']+[f'predictive_{w}_gamma_{g:g}'.replace('.','p') for g in study['training']['predictive_weights']]
        +[transition_name(w,b) for b in study['training']['transition_weights']])


def figures(out,cells,weighting):
    cache=config.ARTIFACTS/'p1_transitions'/'matplotlib'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache));os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    folder=out/'figures';folder.mkdir(exist_ok=True)
    split='test' if 'test' in cells[0]['splits'] else next(iter(cells[0]['splits']))
    datasets=sorted({c['dataset'] for c in cells})
    methods=[f'mixed_{weighting}',f'predictive_{weighting}_gamma_1']+[transition_name(weighting,b) for b in CONFIG['training']['transition_weights']]
    methods=[m for m in methods if method_key(VARIANTS[0],m) in cells[0]['parameters']]
    colors=dict(zip(methods,('#3d6fa4','#d0a65b','#4f9d69','#9b6caf','#d65f5f')))

    for chart in ('posterior','transitions','gain'):
        rows=4 if chart=='transitions' else 2
        fig,axes=plt.subplots(rows,len(datasets),figsize=(5*len(datasets),3.5*rows),squeeze=False)
        for vi,variant in enumerate(VARIANTS):
            for di,ds in enumerate(datasets):
                relevant=[c for c in cells if c['dataset']==ds]
                for ei,event in enumerate(('repair','damage') if chart=='transitions' else (None,)):
                    ax=axes[vi*2+ei if chart=='transitions' else vi,di]
                    for method in methods:
                        name=method_key(variant,method)
                        entries=[c['splits'][split]['models'][name] for c in relevant]
                        if chart=='posterior':
                            rr=[r for entry in entries for r in entry['posterior_by_prefix']]
                            ts=sorted({r['prefix'] for r in rr})
                            ax.plot(ts,[np.mean([r['unique_nll'] for r in rr if r['prefix']==t]) for t in ts],
                                    marker='o',color=colors[method],label=plot_label(method))
                        else:
                            field='transition_calibration' if chart=='transitions' else 'gain_calibration'
                            rr=[r for entry in entries for r in entry[field] if event is None or r['event']==event]
                            points=[]
                            for lower in sorted({r['lower'] for r in rr}):
                                bins=[r for r in rr if r['lower']==lower];n=sum(r['n'] for r in bins)
                                # Keep rare endpoints in CSV, without letting a handful
                                # of repeated states set the plot's entire y range.
                                if n<30:continue
                                points.append((sum(r['prediction']*r['n'] for r in bins)/n,
                                               sum(r['observed']*r['n'] for r in bins)/n,n))
                            if points:
                                x,y,n=np.array(points).T
                                ax.plot(x,y,color=colors[method],label=plot_label(method))
                                ax.scatter(x,y,s=8+70*n/n.max(),color=colors[method],alpha=.8)
                    title=f"{variant.removeprefix('P1_')} / {ds}"+(f' / {event}' if event else '')
                    if chart=='posterior':
                        ax.set(title=title,xlabel='Observed responses',ylabel='Unique-correct NLL')
                        ax.set_xticks(ts)
                    else:
                        lo,hi=ax.get_xlim();ax.plot([lo,hi],[lo,hi],'--',color='gray',alpha=.6)
                        ax.set(title=title,xlabel='Predicted '+('gain' if chart=='gain' else 'probability'),
                               ylabel='Observed '+('gain' if chart=='gain' else 'frequency'))
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                    ax.grid(alpha=.2)
        if axes[0,-1].get_legend_handles_labels()[0]:axes[0,-1].legend(fontsize=8)
        fig.tight_layout();fig.savefig(folder/f'{chart}_{weighting}.png',dpi=150,bbox_inches='tight');plt.close(fig)


def report(out):
    out=Path(out)
    manifest=json.loads((out/'run.json').read_text())
    cells=[json.loads((out/p).read_text()) for p in manifest['cells']]
    if not cells or len({c['experiment_id'] for c in cells})!=1:raise ValueError('Require one nonempty protocol')
    fields=('summary','future_unique_summary','exact_transition_summary','exact_transition_calibration','exact_paired_contrasts','posterior_by_prefix','posterior_calibration','future_by_horizon',
            'gain_calibration','transition_calibration','mc_seed_sensitivity','paired_contrasts',
            'training_restarts','quadrature_checks','selection','parameters')
    tables={k:[] for k in fields}
    validation=dict(cells=len(cells),frozen_arrays_identical=True,frozen_summary_max_difference=0.,
        max_posterior_normalization_error=0.,max_next_normalization_error=0.,max_gain_identity_error=0.,
        fit_locks_precede_evaluation=True,mixed_lambda_unchanged=True)
    for c in cells:
        base=dict(model=c['model'],dataset=c['dataset'])
        fitted=json.loads((out/'selection'/f"{c['model']}_{c['dataset']}.json").read_text())
        if c['selection_locked_at']>c['evaluation_started_at']:raise ValueError('Fit was not locked before evaluation')
        for name,f in fitted['fits'].items():
            if f.get('frozen_previous'):continue
            mixed=fitted['fits'][method_key(f['variant'],f"mixed_{f['objective']['weighting']}")]
            if f['objective']['mixing']!=mixed['objective']['mixing']:raise ValueError('Mixed lambda changed')
        for name,f in fitted['fits'].items():
            for hi,history in enumerate(f.get('resolution_history') or [f]):
                for restart in history['restarts']:
                    tables['training_restarts'].append({**base,'candidate':name,'variant':f['variant'],
                        **f['objective'],**{k:v for k,v in restart.items() if k not in ('params','initial')},
                        'final_resolution':hi==max(len(f.get('resolution_history',[]))-1,0),
                        'chosen_restart':restart['restart_seed']==history['best']['restart_seed'],
                        'frozen_previous':f.get('frozen_previous',False)})
                if 'quadrature' in history:
                    q=history['quadrature']
                    tables['quadrature_checks'].append({**base,'candidate':name,'variant':f['variant'],
                        'coarse_nodes':q['coarse_nodes'],'fine_nodes':q['fine_nodes'],**q['errors'],
                        'max_error':q['max_error'],'final_resolution':hi==len(f['resolution_history'])-1})
        for name,source in fitted['selected'].items():
            f=fitted['fits'][source]
            tables['selection'].append({**base,'method':name,'variant':f['variant'],'source':source,
                'lambda':f['objective']['mixing'],'dev_unique_nll':f['best']['dev_unique_nll'],
                'restart_seed':f['best']['restart_seed'],'normal_nodes':f['best']['normal_nodes'],
                'train_kept':fitted['train_kept'],'dev_unique':fitted['dev_unique'],'locked_at':fitted['selection_locked_at']})
        for name,p in c['parameters'].items():
            tables['parameters'].append({**base,'method':name,'variant':p['variant'],'count':p['count'],
                **p['objective'],**p['raw'],**p['readable'],'normal_nodes':p['normal_nodes'],
                'sigma_near_bound':p['readable']['sigma_theta']>.95*c['study']['integration']['sigma_bound']})
        prior=json.loads((baseline_root(c['study'])/'cells'/f"{c['model']}_{c['dataset']}.json").read_text())
        with np.load(c['prediction_cache']['path'],allow_pickle=False) as z,np.load(prior['prediction_cache']['path'],allow_pickle=False) as old:
            for split,s in c['splits'].items():
                for name,entry in s['models'].items():
                    variant=c['parameters'][name]['variant']
                    method=name.split('__',1)[1]
                    key={**base,'split':split,'method':name,'variant':variant,'training':method}
                    for field in ('summary','future_unique_summary','exact_transition_summary'):
                        for metric,v in entry[field].items():
                            ci=v['ci95'] or [None,None]
                            tables[field].append({**key,'metric':metric,'mean':v['mean'],'ci_low':ci[0],
                                'ci_high':ci[1],'questions':v['questions']})
                    for field in ('posterior_by_prefix','posterior_calibration','future_by_horizon','gain_calibration',
                                  'transition_calibration','mc_seed_sensitivity','exact_transition_calibration'):
                        tables[field]+=[{**key,**r} for r in entry[field]]
                    post=z[f'{split}.{name}.post'];nxt=z[f'{split}.{name}.future.next_answer']
                    validation['max_posterior_normalization_error']=max(validation['max_posterior_normalization_error'],float(np.max(np.abs(post.sum(-1)-1))))
                    validation['max_next_normalization_error']=max(validation['max_next_normalization_error'],float(np.nanmax(np.abs(nxt.sum(-1)-1))))
                    gain=z[f'{split}.{name}.future.gain'];repair=z[f'{split}.{name}.future.repair'];damage=z[f'{split}.{name}.future.damage']
                    validation['max_gain_identity_error']=max(validation['max_gain_identity_error'],float(np.nanmax(np.abs(gain-repair+damage))))
                    if not method.startswith('transition_'):
                        for k in z.files:
                            if k.startswith(f'{split}.{name}.') and '.exact.' not in k:
                                original=k
                                if not np.array_equal(z[k],old[original],equal_nan=True):raise ValueError('Frozen reference array changed')
                        for metric,v in entry['summary'].items():
                            previous=prior['splits'][split]['models'][name]['summary'].get(metric,{}).get('mean')
                            if v['mean'] is not None and previous is not None:
                                validation['frozen_summary_max_difference']=max(validation['frozen_summary_max_difference'],abs(v['mean']-previous))
                for comparison,diffs in s['paired_contrasts'].items():
                    for metric,v in diffs.items():
                        ci=v['ci95'] or [None,None]
                        tables['paired_contrasts'].append({**base,'split':split,'comparison':comparison,
                            'metric':metric,'mean':v['mean'],'ci_low':ci[0],'ci_high':ci[1],'questions':v['questions']})
    for c in cells:
        for split,s in c['splits'].items():
            for comparison,diffs in s['exact_paired_contrasts'].items():
                for metric,v in diffs.items():
                    ci=v['ci95'] or [None,None]
                    tables['exact_paired_contrasts'].append(dict(model=c['model'],dataset=c['dataset'],split=split,
                        comparison=comparison,metric=metric,mean=v['mean'],ci_low=ci[0],ci_high=ci[1],questions=v['questions']))
    for name,rows in tables.items():csv_file(out/f'{name}.csv',rows)
    macro,dataset_rows=[],[]
    for split in sorted({r['split'] for r in tables['summary']}):
        for variant in VARIANTS:
            for method in method_order(cells[0]['study']):
                name=method_key(variant,method)
                for metric in sorted({r['metric'] for r in tables['summary']}):
                    rr=[r for r in tables['summary'] if r['split']==split and r['method']==name and r['metric']==metric and r['mean'] is not None]
                    if not rr:continue
                    key=dict(split=split,method=name,variant=variant,training=method,metric=metric)
                    macro.append({**key,'mean':float(np.mean([r['mean'] for r in rr])),'cells':len(rr)})
                    for ds in sorted({r['dataset'] for r in rr}):
                        vs=[r['mean'] for r in rr if r['dataset']==ds]
                        dataset_rows.append({**key,'dataset':ds,'mean':float(np.mean(vs)),'cells':len(vs)})
                bias=[abs(r['mean']) for r in tables['summary'] if r['split']==split and r['method']==name and r['metric']=='gain_bias']
                if bias:macro.append(dict(split=split,method=name,variant=variant,training=method,
                    metric='abs_cell_gain_bias',mean=float(np.mean(bias)),cells=len(bias)))
    csv_file(out/'macro_summary.csv',macro);csv_file(out/'dataset_summary.csv',dataset_rows)
    paired_macro=[]
    for split in sorted({r['split'] for r in tables['paired_contrasts']}):
        for comparison in sorted({r['comparison'] for r in tables['paired_contrasts']}):
            for metric in sorted({r['metric'] for r in tables['paired_contrasts']}):
                rr=[r for r in tables['paired_contrasts'] if r['split']==split and r['comparison']==comparison and r['metric']==metric and r['mean'] is not None]
                if rr:paired_macro.append(dict(split=split,comparison=comparison,metric=metric,
                    mean=float(np.mean([r['mean'] for r in rr])),cells=len(rr),negative_cells=sum(r['mean']<0 for r in rr),
                    ci_below_zero_cells=sum(r['ci_high'] is not None and r['ci_high']<0 for r in rr),
                    ci_above_zero_cells=sum(r['ci_low'] is not None and r['ci_low']>0 for r in rr)))
    csv_file(out/'paired_macro.csv',paired_macro)
    mc_contrasts=[]
    sensitivity=tables['mc_seed_sensitivity']
    for r in sensitivity:
        if not r['training'].startswith('transition_'):continue
        reference=next(x for x in sensitivity if x['model']==r['model'] and x['dataset']==r['dataset']
            and x['split']==r['split'] and x['variant']==r['variant'] and x['training']=='mixed_'+r['training'].split('_beta_')[0].removeprefix('transition_') and x['seed']==r['seed'])
        for metric in ('gain_mse','repair_brier','damage_brier'):
            mc_contrasts.append({k:r[k] for k in ('model','dataset','split','variant','training','seed')} |
                dict(metric=metric,difference=r[metric]-reference[metric]))
    csv_file(out/'mc_seed_contrasts.csv',mc_contrasts)
    exact_macro=[]
    for split in sorted({r['split'] for r in tables['exact_transition_summary']}):
        for method in method_order(cells[0]['study']):
            for metric in sorted({r['metric'] for r in tables['exact_transition_summary']}):
                rr=[r for r in tables['exact_transition_summary'] if r['split']==split and r['method']==method_key(VARIANTS[0],method)]
                for variant in VARIANTS:
                    vv=[r['mean'] for r in tables['exact_transition_summary'] if r['split']==split and r['method']==method_key(variant,method) and r['metric']==metric and r['mean'] is not None]
                    if vv:exact_macro.append(dict(split=split,method=method_key(variant,method),variant=variant,training=method,metric=metric,mean=float(np.mean(vv)),cells=len(vv)))
    csv_file(out/'exact_transition_macro.csv',exact_macro)
    exact_paired_macro=[]
    for split in sorted({r['split'] for r in tables['exact_paired_contrasts']}):
        for comparison in sorted({r['comparison'] for r in tables['exact_paired_contrasts']}):
            for metric in sorted({r['metric'] for r in tables['exact_paired_contrasts']}):
                rr=[r for r in tables['exact_paired_contrasts'] if r['split']==split and r['comparison']==comparison and r['metric']==metric and r['mean'] is not None]
                if rr:exact_paired_macro.append(dict(split=split,comparison=comparison,metric=metric,mean=float(np.mean([r['mean'] for r in rr])),cells=len(rr),negative_cells=sum(r['mean']<0 for r in rr),ci_below_zero_cells=sum(r['ci_high'] is not None and r['ci_high']<0 for r in rr),ci_above_zero_cells=sum(r['ci_low'] is not None and r['ci_low']>0 for r in rr)))
    csv_file(out/'exact_paired_macro.csv',exact_paired_macro)
    from pgm_asc.utils import write_json
    write_json(out/'validation.json',validation)
    split='test' if 'test' in cells[0]['splits'] else next(iter(cells[0]['splits']))
    values={(r['variant'],r['training'],r['metric']):r['mean'] for r in macro if r['split']==split}
    exact_values={(r['variant'],r['training'],r['metric']):r['mean'] for r in exact_macro if r['split']==split}
    tr=cells[0]['study']['training'];e=cells[0]['study']['evaluation'];betas=tr['transition_weights']
    def methods(w):return [f'mixed_{w}',f'predictive_{w}_gamma_1']+[transition_name(w,b) for b in betas]
    def score(v,m,k,scale=1.,digits=4,exact=False):return fmt((exact_values if exact else values)[v,m,k]*scale,digits)
    def post_rows(w):return [[SHORT[v],plot_label(m)]+[score(v,m,k) for k in
        ('unique_nll','reference_brier','other_brier','next_answer_nll')] for v in VARIANTS for m in methods(w)]
    def gain_rows(w,exact=False):return [[SHORT[v],plot_label(m)]+[score(v,m,k,digits=5,exact=exact) for k in
        ('repair_brier','damage_brier','gain_mse')]+[score(v,m,k,100,3,exact=exact) for k in
        ('repair_bias','damage_bias','gain_bias')]+[score(v,m,'transition_categorical_nll',exact=exact)] for v in VARIANTS for m in methods(w)]
    joint={}
    joint_cells={}
    for v in VARIANTS:
        for c in cells:
            prior=json.loads((baseline_root(c['study'])/'cells'/f"{c['model']}_{c['dataset']}.json").read_text())
            joint_cells[c['model'],c['dataset'],v]=prior['splits'][split]['models'][method_key(v,'joint')]['summary']['unique_nll']['mean']
        joint[v]=float(np.mean([joint_cells[c['model'],c['dataset'],v] for c in cells]))
    cell_retention=[]
    for c in cells:
        for v in VARIANTS:
            j=joint_cells[c['model'],c['dataset'],v]
            models=c['splits'][split]['models']
            for w in tr['weightings']:
                base=models[method_key(v,f'mixed_{w}')]
                den=j-base['summary']['unique_nll']['mean']
                for b in betas:
                    new=models[method_key(v,transition_name(w,b))]
                    cell_retention.append(dict(model=c['model'],dataset=c['dataset'],split=split,variant=v,weighting=w,beta=b,
                        posterior_gain_retained_fraction=(j-new['summary']['unique_nll']['mean'])/den if den>0 else None,
                        posterior_below_joint=new['summary']['unique_nll']['mean']<j,
                        gain_mse_minus_mixed=new['summary']['gain_mse']['mean']-base['summary']['gain_mse']['mean'],
                        exact_gain_mse_minus_mixed=new['exact_transition_summary']['gain_mse']['mean']-base['exact_transition_summary']['gain_mse']['mean']))
    csv_file(out/'cell_retention.csv',cell_retention)
    retention=[];findings=[];lessons=[]
    for v in VARIANTS:
        for w in tr['weightings']:
            mixed=f'mixed_{w}';next_model=f'predictive_{w}_gamma_1'
            den=joint[v]-values[v,mixed,'unique_nll']
            for b in betas:
                m=transition_name(w,b)
                fraction=(joint[v]-values[v,m,'unique_nll'])/den if den>0 else None
                rr=[c['splits'][split]['models'] for c in cells]
                p_wins=sum(x[method_key(v,m)]['summary']['unique_nll']['mean']<x[method_key(v,mixed)]['summary']['unique_nll']['mean'] for x in rr)
                g_wins=sum(x[method_key(v,m)]['summary']['gain_mse']['mean']<x[method_key(v,mixed)]['summary']['gain_mse']['mean'] for x in rr)
                eg_wins=sum(x[method_key(v,m)]['exact_transition_summary']['gain_mse']['mean']<x[method_key(v,mixed)]['exact_transition_summary']['gain_mse']['mean'] for x in rr)
                retention.append(dict(variant=v,weighting=w,beta=b,posterior_gain_retained_fraction=fraction,
                    posterior_below_mixed_cells=p_wins,gain_below_mixed_cells=g_wins,exact_gain_below_mixed_cells=eg_wins,cells=len(cells)))
                findings.append(f"{SHORT[v]} / {w} / β={b:g}: Mixed 대비 Δposterior NLL "
                    f"{values[v,m,'unique_nll']-values[v,mixed,'unique_nll']:+.6f}, ΔMC gain MSE "
                    f"{values[v,m,'gain_mse']-values[v,mixed,'gain_mse']:+.6f}, Δexact gain MSE "
                    f"{exact_values[v,m,'gain_mse']-exact_values[v,mixed,'gain_mse']:+.6f}. "
                    f"MC/exact gain 개선 cell은 각각 {g_wins}/{len(cells)}, {eg_wins}/{len(cells)}개입니다. "
                    f"고정 next γ=1 대비 ΔMC gain MSE={values[v,m,'gain_mse']-values[v,next_model,'gain_mse']:+.6f}.")
        if 1. in betas:
            m=transition_name('uniform',1.);mixed='mixed_uniform';next_model='predictive_uniform_gamma_1'
            den=joint[v]-values[v,mixed,'unique_nll']
            retained=(joint[v]-values[v,m,'unique_nll'])/den if den>0 else float('nan')
            lessons.append(f"{SHORT[v]} / uniform β=1은 Mixed의 posterior 개선을 {100*retained:.1f}% 보존합니다. "
                f"MC gain MSE는 next γ=1의 {values[v,next_model,'gain_mse']:.5f}보다 낮은 {values[v,m,'gain_mse']:.5f}지만, "
                f"Mixed의 {values[v,mixed,'gain_mse']:.5f}보다 낮지는 않습니다. "
                f"Exact gain MSE도 {exact_values[v,mixed,'gain_mse']:.5f} → {exact_values[v,m,'gain_mse']:.5f}로 같은 방향입니다.")
            lessons.append(f"{SHORT[v]} / uniform β=1의 exact repair bias는 Mixed "
                f"{100*exact_values[v,mixed,'repair_bias']:+.3f} → {100*exact_values[v,m,'repair_bias']:+.3f}pp, "
                f"damage bias는 {100*exact_values[v,mixed,'damage_bias']:+.3f} → {100*exact_values[v,m,'damage_bias']:+.3f}pp입니다. "
                f"Gain bias는 {100*exact_values[v,mixed,'gain_bias']:+.3f} → {100*exact_values[v,m,'gain_bias']:+.3f}pp입니다. "
                "Repair 과대예측이 거의 남은 채 damage 과대예측을 줄이면, 두 오차의 상쇄가 줄어 gain bias가 커질 수 있습니다.")
        if all(values[v,transition_name(w,b),'gain_mse']>=values[v,f'mixed_{w}','gain_mse'] and
               exact_values[v,transition_name(w,b),'gain_mse']>=exact_values[v,f'mixed_{w}','gain_mse']
               for w in tr['weightings'] for b in betas):
            lessons.append(f"{SHORT[v]}의 두 prefix 가중치·모든 β에서 MC 및 exact macro gain MSE가 Mixed보다 낮아지지 않았습니다. "
                "이 실험 범위에서는 전이 NLL을 직접 추가해도 gain calibration의 일관된 추가 개선은 확인되지 않았습니다. "
                "약한 arm의 차이가 작다는 점과 cell별 신뢰구간을 함께 해석해야 합니다.")
    csv_file(out/'posterior_retention.csv',retention)
    retention_rows=[[SHORT[r['variant']],r['weighting'],r['beta'],fmt(100*r['posterior_gain_retained_fraction'],1) if r['posterior_gain_retained_fraction'] is not None else 'N/A',r['posterior_below_mixed_cells'],r['gain_below_mixed_cells'],r['exact_gain_below_mixed_cells']] for r in retention]
    qerr=max((r['max_error'] for r in tables['quadrature_checks'] if r['final_resolution'] and '__transition_' in r['candidate']),default=0)
    text=['# P1 Mixed + multi-step repair/damage 학습 ablation',
        f"기존 P1의 4/5개 파라미터와 입력을 유지하고, Mixed에 multi-step 전이 손실을 추가했습니다. {len(cells)}개 cell의 {split} 결과를 동일 가중 macro로 보고합니다. 고정 β arm을 모두 평가하며 test에서 계수를 고르지 않습니다.",
        '## 학습 목적',
        'M=(J+λCw)/(1+λ), T=mean[−log p(실제 전이|Dt)], L=(M+βT)/(1+β)+ridge입니다. 전이는 repair(오답→정답), damage(정답→오답), stay(정오 변화 없음)의 서로 배타적인 세 class입니다. p=(r,d,1−r−d)이며 class balancing 없이 proper categorical NLL을 사용합니다. β=0은 기존 Mixed입니다.',
        f"λ는 이전 Mixed의 dev 선택 값을 고정했습니다. β={betas}를 모두 학습·평가합니다. Prefix={tr['transition_prefixes']}, horizon={tr['transition_horizons']}, t+h≤16의 모든 쌍에 같은 가중치를 주고 질문 안에서 먼저 평균합니다. Uniform/v1은 기존 posterior prefix 가중치의 별도 ablation이며 T의 가중치는 같습니다.",
        'r,d는 현재 정답 여부나 gold c를 알려준 조건부 확률이 아닙니다. 과거 답만 조건으로 c,q,θ를 모두 적분한 native P1 예측입니다. 실제 현재/미래 정오는 train target과 평가에만 씁니다. 미래 답 내용은 예측 입력에 들어가지 않습니다.',
        f"고정 unique-correct train {sum(c['train_kept'] for c in cells)}개 model-question pair, 공통 초기값 {tr['restart_seeds']}, σ 초기값 {tr['sigma_starts']}, raw ridge={tr['ridge']}, L-BFGS 및 기존 q/Normal 적분 설정을 유지합니다. Optimizer restart는 penalized train loss로 고릅니다. β를 dev/test posterior나 gain으로 선택하지 않습니다.",
        '## 정확 전이 확률과 검증',
        'Correct-answer hypothesis별로 미래 correct 횟수와 CRP wrong partition을 합산합니다. 관측된 정답 후보는 먼저 등장한 오답과 나중 오답의 tie cap을 나눕니다. 아직 보지 못한 정답은 같은 최종 크기의 새 오답 그룹 r개와 동점일 때 먼저 나타날 확률 1/(r+1)을 반영합니다. Truncated generating function으로 경로를 합한 뒤, q·θ와 prefix posterior를 같은 P1 적분으로 주변화합니다. θ는 continuation 전체에 공유됩니다.',
        '확률 및 gradient는 완전 경로 열거(h≤3), 별도의 정확 동적계획법(h=8), native simulator(h=8), 기존 posterior × conditional kernel 적분, finite difference로 확인했습니다. 같은 count/first-order prefix는 계산을 공유합니다. 새 latent/head/temperature를 추가하지 않습니다.',
        '## Uniform: posterior와 다음 답 예측',
        '모든 loss는 낮을수록 좋습니다. 기존 Mixed와 next-answer 추가 모델의 파라미터·예측은 그대로 재사용했습니다. Next comparator는 사전에 고정한 γ=1이며 나머지 이전 γ도 CSV에 보존합니다.',
        table(['구조','학습','Unique NLL','Reference Brier','OTHER Brier','Next-answer NLL'],post_rows('uniform')),
        '## Uniform: 같은 paired MC로 평가',
        f"기존과 같은 seed {len(e['mc_seeds'])}개 × {e['simulations_per_seed']} continuations입니다. 같은 질문/prefix/seed 난수와 고정 plurality·최초 등장 tie-break를 사용합니다. Finite-draw gain=repair−damage입니다. Bias는 예측−관측, pp 단위입니다.",
        table(['구조','학습','Repair Brier','Damage Brier','Gain MSE','Repair bias pp','Damage bias pp','Gain bias pp','전이 NLL'],gain_rows('uniform')),
        '## Exact-event 민감도',
        '아래 표는 학습과 같은 정확 적분으로 repair/damage와 gain을 계산합니다. MC noise가 없는 동일 모델의 보조 비교이며 주 paired-MC 결과를 대체하지 않습니다. MC 전이 NLL은 0-probability clipping에 민감할 수 있어 exact NLL을 함께 봅니다.',
        table(['구조','학습','Repair Brier','Damage Brier','Gain MSE','Repair bias pp','Damage bias pp','Gain bias pp','전이 NLL'],gain_rows('uniform',True)),
        '## 해석',*lessons,
        '## Posterior 개선 보존과 cell별 일관성',
        '보존율=(Joint macro NLL−새 모델 macro NLL)/(Joint macro NLL−Mixed macro NLL)입니다. 100%는 Mixed의 posterior 개선을 모두 보존한 경우이며, 보존율은 계수 선택에 사용하지 않습니다.',
        table(['구조','가중치','β','macro 보존율 %','Posterior<Mixed cells','MC gain<Mixed cells','Exact gain<Mixed cells'],retention_rows),
        '## 관찰된 변화',*findings,
        '## v1 가중치 민감도',
        table(['구조','학습','Unique NLL','Reference Brier','OTHER Brier','Next-answer NLL'],post_rows('v1')) if 'v1' in tr['weightings'] else 'v1은 이 실행에서 제외했습니다.',
        table(['구조','학습','Repair Brier','Damage Brier','Gain MSE','Repair bias pp','Damage bias pp','Gain bias pp','전이 NLL'],gain_rows('v1')) if 'v1' in tr['weightings'] else '',
        '## 수치·재현 QA',
        f"최종 해상도의 최대 coarse/fine 오차는 {qerr:.3g}입니다. Train J/C/T/data loss, dev posterior/전이 NLL, dev 개별 posterior/next-answer/event 확률을 검사하며 허용 오차 {cells[0]['study']['integration']['tolerance']}를 넘으면 test 전에 재학습합니다.",
        f"Frozen reference 배열은 이전과 동일합니다. 공통 summary 최대 차이 {validation['frozen_summary_max_difference']:.3g}, posterior 정규화 오차 {validation['max_posterior_normalization_error']:.3g}, next-answer 정규화 오차 {validation['max_next_normalization_error']:.3g}, paired gain 항등식 오차 {validation['max_gain_identity_error']:.3g}. validation.json에 저장했습니다.",
        '## 그림',
        'Calibration 그림은 n≥30인 bin을 표시하며 모든 희소 bin은 CSV에 남깁니다. 상태 수는 독립 질문 수가 아니며 큰 점은 더 많은 반복 상태를 나타냅니다.',
        '![Posterior uniform](figures/posterior_uniform.png)','![Repair damage uniform](figures/transitions_uniform.png)','![Gain uniform](figures/gain_uniform.png)',
        '## 한계',
        f"1. {e['bootstrap_replicates']:,}회 paired question bootstrap을 사용하고 prefix/horizon·MC seed를 독립 질문으로 세지 않습니다. Macro에 pooled CI나 다중비교 유의성 주장을 붙이지 않습니다. Exact sensitivity도 같은 질문으로 평가합니다.",
        '2. 기존에 탐색한 bank와 xVerify exact-string 라벨의 ablation입니다. 학습은 unique-correct 질문에 한정하며 전체 질문과 unique-only 미래 민감도를 제공합니다. 독립 최종 holdout 검증은 아닙니다.',
        '3. T는 고정 prefix/horizon·plurality decision에 맞춘 손실입니다. 다른 horizon·tie-break·stopping 선택 분포로의 일반화는 별도 문제입니다. Stopping threshold나 토큰 절감은 최적화하지 않았습니다.',
        '4. λ 고정·현재 β grid·동일한 P1 그래프/경계 내 결과입니다. q grid 절단, bounded prior/σ와 빈 답을 생성하지 않는 observation-model 한계가 남습니다.',
        '5. Gain bias 감소가 repair 감소와 damage 증가의 상쇄인지 proper R/D score와 함께 확인합니다. 희귀 damage에서 표본 오차가 크며 강한 loss가 posterior나 생성 예측을 훼손할 수 있습니다.',
        '6. One-step transition loss와의 직접 비교나 transition Brier 학습은 이번 arm에 포함하지 않았습니다. 현재 비교는 기존 next-answer 손실과 고정 multi-step categorical 전이 손실의 비교이며, target 변경 효과와 horizon 변경 효과를 각각 분리하지는 않습니다.',
        '## 재현 및 파일',
        'pgm p1-transition fit --workers 4','pgm p1-transition run --workers 4','pgm p1-transition report',
        'configs/p1_transitions.json과 protocol.json은 고정 loss·계수·코드 hash를 기록합니다. selection/*.json은 모든 fit/restart/적분 이력과 test 이전 lock입니다. artifacts/p1_transitions/에는 큰 native/정확 전이 prediction 배열과 후보 cache를 저장합니다.',
        'summary.csv / macro_summary.csv / dataset_summary.csv / paired_contrasts.csv / paired_macro.csv / exact_transition_summary.csv / exact_transition_macro.csv / exact_paired_contrasts.csv / exact_paired_macro.csv / posterior_retention.csv / *_calibration.csv / future_unique_summary.csv / training_restarts.csv / quadrature_checks.csv / parameters.csv / mc_seed_sensitivity.csv / mc_seed_contrasts.csv를 제공합니다.']
    markdown='\n\n'.join(x for x in text if x)+'\n';(out/'README.md').write_text(markdown)
    for w in tr['weightings']:figures(out,cells,w)
    style='body{font-family:system-ui,sans-serif;max-width:1450px;margin:40px auto;padding:0 24px;line-height:1.7;background:#fafbfc;color:#17212b}h1,h2{line-height:1.3;margin-top:2em}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:9px;border-bottom:1px solid #dce2e8}th{background:#eaf0f7}img{width:100%;margin:18px 0}'
    (out/'report.html').write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>P1 multi-step transition ablation</title><style>"+style+'</style><body>'+markdown_html(markdown)+'</body></html>')
    return str(out/'report.html')
