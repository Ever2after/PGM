"""Repair/damage Brier ablation: frozen Mixed/NLL references and exact sensitivity."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from pgm_asc import config
from pgm_asc.diagnostics.report import csv_file,fmt,markdown_html,table
from pgm_asc.p1_brier.run import baseline_root,method_key,brier_name
from pgm_asc.p1_transitions.run import baseline_root as nll_source_root
from pgm_asc.p1_brier import CONFIG

VARIANTS=('P1_independent','P1_linked')
SHORT={'P1_independent':'q 연결 제거','P1_linked':'q 연결 유지'}
def plot_label(method):
    if method.startswith('brier_'):
        return 'Mixed + R/D Brier, beta='+method.split('_beta_')[1].replace('p','.')
    if method.startswith('transition_'):
        return 'Mixed + R/D NLL, beta='+method.split('_beta_')[1].replace('p','.')
    return 'Mixed'


def method_order(study):
    return tuple(m for w in study['training']['weightings'] for m in [f'mixed_{w}']
        +[f'transition_{w}_beta_{b:g}'.replace('.','p') for b in study['training']['transition_weights']]
        +[brier_name(w,b) for b in study['training']['brier_weights']])


def figures(out,cells,weighting):
    cache=config.ARTIFACTS/'p1_brier'/'matplotlib'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache));os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    folder=out/'figures';folder.mkdir(exist_ok=True)
    split='test' if 'test' in cells[0]['splits'] else next(iter(cells[0]['splits']))
    datasets=sorted({c['dataset'] for c in cells})
    methods=[f'mixed_{weighting}',f'transition_{weighting}_beta_1']+[brier_name(weighting,b) for b in CONFIG['training']['brier_weights']]
    methods=[m for m in methods if method_key(VARIANTS[0],m) in cells[0]['parameters']]
    colors=dict(zip(methods,('#3d6fa4','#d0a65b','#4f9d69','#9b6caf','#d65f5f','#666666')))

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
        fig.tight_layout();fig.savefig(folder/f'{chart}_{weighting}.png',dpi=150,bbox_inches='tight',pad_inches=.3);plt.close(fig)


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
                    if not method.startswith('brier_'):
                        for k in z.files:
                            if k.startswith(f'{split}.{name}.'):
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
                calibration=[r['calibration_error'] for r in tables['future_by_horizon'] if r['split']==split
                    and r['method']==name and r['event']=='gain']
                if calibration:macro.append(dict(split=split,method=name,variant=variant,training=method,
                    metric='gain_ece',mean=float(np.mean(calibration)),cells=len(calibration)//len({(r['prefix'],r['horizon']) for r in tables['future_by_horizon'] if r['event']=='gain'})))
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
        if not r['training'].startswith('brier_'):continue
        reference=next(x for x in sensitivity if x['model']==r['model'] and x['dataset']==r['dataset']
            and x['split']==r['split'] and x['variant']==r['variant'] and x['training']=='mixed_'+r['training'].split('_beta_')[0].removeprefix('brier_') and x['seed']==r['seed'])
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
    tr=cells[0]['study']['training'];e=cells[0]['study']['evaluation'];betas=tr['brier_weights']
    def methods(w):return [f'mixed_{w}',f'transition_{w}_beta_1']+[brier_name(w,b) for b in betas]
    def score(v,m,k,scale=1.,digits=4,exact=False):return fmt((exact_values if exact else values)[v,m,k]*scale,digits)
    def post_rows(w):return [[SHORT[v],plot_label(m)]+[score(v,m,k) for k in
        ('unique_nll','reference_brier','other_brier','next_answer_nll')] for v in VARIANTS for m in methods(w)]
    def gain_rows(w,exact=False):return [[SHORT[v],plot_label(m)]+[score(v,m,k,digits=5,exact=exact) for k in
        ('repair_brier','damage_brier','rd_mean_brier','gain_mse')]+[score(v,m,k,100,3,exact=exact) for k in
        ('repair_bias','damage_bias','gain_bias')]+([] if exact else [score(v,m,'gain_ece',100,3)]) for v in VARIANTS for m in methods(w)]
    joint_cells={};joint={}
    for c in cells:
        previous=json.loads((baseline_root(c['study'])/'cells'/f"{c['model']}_{c['dataset']}.json").read_text())
        joint_path=nll_source_root(previous['study'])/'cells'/f"{c['model']}_{c['dataset']}.json"
        prior=json.loads(joint_path.read_text())
        for v in VARIANTS:joint_cells[c['model'],c['dataset'],v]=prior['splits'][split]['models'][method_key(v,'joint')]['summary']['unique_nll']['mean']
    for v in VARIANTS:joint[v]=float(np.mean([joint_cells[c['model'],c['dataset'],v] for c in cells]))
    retention=[];cell_retention=[];findings=[]
    for v in VARIANTS:
        for w in tr['weightings']:
            mixed=f'mixed_{w}';nll=f'transition_{w}_beta_1';den=joint[v]-values[v,mixed,'unique_nll']
            for b in betas:
                m=brier_name(w,b);fraction=(joint[v]-values[v,m,'unique_nll'])/den if den>0 else None
                p=g=eg=bg=0
                for c in cells:
                    models=c['splits'][split]['models'];new=models[method_key(v,m)];base=models[method_key(v,mixed)]
                    j=joint_cells[c['model'],c['dataset'],v];d=j-base['summary']['unique_nll']['mean']
                    p+=new['summary']['unique_nll']['mean']<j
                    g+=new['summary']['gain_mse']['mean']<base['summary']['gain_mse']['mean']
                    eg+=new['exact_transition_summary']['gain_mse']['mean']<base['exact_transition_summary']['gain_mse']['mean']
                    bg+=new['exact_transition_summary']['rd_mean_brier']['mean']<base['exact_transition_summary']['rd_mean_brier']['mean']
                    cell_retention.append(dict(model=c['model'],dataset=c['dataset'],split=split,variant=v,weighting=w,beta=b,
                        posterior_gain_retained_fraction=(j-new['summary']['unique_nll']['mean'])/d if d>0 else None,
                        posterior_nll_minus_mixed=new['summary']['unique_nll']['mean']-base['summary']['unique_nll']['mean'],
                        gain_mse_minus_mixed=new['summary']['gain_mse']['mean']-base['summary']['gain_mse']['mean'],
                        exact_gain_mse_minus_mixed=new['exact_transition_summary']['gain_mse']['mean']-base['exact_transition_summary']['gain_mse']['mean']))
                retention.append(dict(variant=v,weighting=w,beta=b,posterior_gain_retained_fraction=fraction,
                    posterior_below_joint_cells=p,mc_gain_below_mixed_cells=g,exact_gain_below_mixed_cells=eg,exact_brier_below_mixed_cells=bg,cells=len(cells)))
                findings.append(f"{SHORT[v]} / {w} / β={b:g}: Mixed 대비 Δposterior NLL "
                    f"{values[v,m,'unique_nll']-values[v,mixed,'unique_nll']:+.6f}, ΔRD mean Brier "
                    f"{values[v,m,'rd_mean_brier']-values[v,mixed,'rd_mean_brier']:+.6f}, ΔMC gain MSE "
                    f"{values[v,m,'gain_mse']-values[v,mixed,'gain_mse']:+.6f}, Δexact gain MSE "
                    f"{exact_values[v,m,'gain_mse']-exact_values[v,mixed,'gain_mse']:+.6f}. "
                    f"MC/exact gain 개선 cell은 {g}/{len(cells)}, {eg}/{len(cells)}개입니다. "
                    f"기존 NLL β=1 대비 ΔMC gain MSE={values[v,m,'gain_mse']-values[v,nll,'gain_mse']:+.6f}.")
    csv_file(out/'posterior_retention.csv',retention);csv_file(out/'cell_retention.csv',cell_retention)
    decomposition=[]
    for c in cells:
        for split_name,s in c['splits'].items():
            for name,entry in s['models'].items():
                for field,mode in (('summary','paired_mc'),('exact_transition_summary','exact_event')):
                    d={k:v['mean'] for k,v in entry[field].items()}
                    decomposition.append(dict(model=c['model'],dataset=c['dataset'],split=split_name,method=name,mode=mode,
                        repair_brier=d['repair_brier'],damage_brier=d['damage_brier'],rd_mean_brier=d['rd_mean_brier'],
                        error_cross=d['rd_error_cross'],gain_mse=d['gain_mse'],identity_residual=d['gain_decomposition_residual']))
    csv_file(out/'gain_decomposition.csv',decomposition)
    decomp_macro=[]
    for v in VARIANTS:
        for method in method_order(cells[0]['study']):
            name=method_key(v,method)
            for mode in ('paired_mc','exact_event'):
                rr=[r for r in decomposition if r['split']==split and r['method']==name and r['mode']==mode]
                if rr:decomp_macro.append(dict(variant=v,training=method,split=split,mode=mode,cells=len(rr),
                    **{k:float(np.mean([r[k] for r in rr])) for k in ('repair_brier','damage_brier','rd_mean_brier','error_cross','gain_mse','identity_residual')}))
    csv_file(out/'gain_decomposition_macro.csv',decomp_macro)
    retained_rows=[[SHORT[r['variant']],r['weighting'],r['beta'],fmt(100*r['posterior_gain_retained_fraction'],1) if r['posterior_gain_retained_fraction'] is not None else 'N/A',r['posterior_below_joint_cells'],r['mc_gain_below_mixed_cells'],r['exact_gain_below_mixed_cells']] for r in retention]
    conclusions=[]
    for v in VARIANTS:
        for w in tr['weightings']:
            mixed=f'mixed_{w}';nll=f'transition_{w}_beta_1'
            mc_better=[b for b in betas if values[v,brier_name(w,b),'gain_mse']<values[v,mixed,'gain_mse']]
            exact_better=[b for b in betas if exact_values[v,brier_name(w,b),'gain_mse']<exact_values[v,mixed,'gain_mse']]
            if not mc_better and not exact_better:
                result='어떤 β도 Mixed의 macro gain MSE를 낮추지 못했습니다. Paired MC와 exact-event 평가의 방향이 같습니다.'
            else:
                result=f'Mixed보다 macro gain MSE가 낮은 β는 paired MC {mc_better}, exact-event {exact_better}입니다. 이는 계수 선택이나 유의성 판정이 아닌 관측된 점추정 비교입니다.'
            conclusions.append(f'{SHORT[v]} / {w}: {result}')
            if 1 in betas:
                m=brier_name(w,1);r=next(r for r in retention if r['variant']==v and r['weighting']==w and r['beta']==1)
                preserved=fmt(100*r['posterior_gain_retained_fraction'],1) if r['posterior_gain_retained_fraction'] is not None else 'N/A'
                conclusions.append(f"β=1은 Mixed의 posterior NLL 개선을 {preserved}% 보존합니다. "
                    f"Mixed 대비 Δnext-answer NLL={values[v,m,'next_answer_nll']-values[v,mixed,'next_answer_nll']:+.6f}, "
                    f"Δgain ECE={100*(values[v,m,'gain_ece']-values[v,mixed,'gain_ece']):+.4f} pp입니다. "
                    f"기존 전이 NLL β=1 대비 Δposterior NLL={values[v,m,'unique_nll']-values[v,nll,'unique_nll']:+.6f}, "
                    f"Δgain MSE={values[v,m,'gain_mse']-values[v,nll,'gain_mse']:+.6f}. "
                    '두 손실은 단위가 달라 같은 β만으로 최적 학습 목적의 우열을 단정할 수 없습니다.')
    exact_not_better=all(exact_values[v,brier_name(w,b),'gain_mse']>=exact_values[v,f'mixed_{w}','gain_mse']
        for v in VARIANTS for w in tr['weightings'] for b in betas)
    if exact_not_better:
        conclusions.insert(0,'이번 고정 grid에서는 posterior 개선을 대부분 유지하는 작은 β를 찾았지만 gain calibration의 일관된 추가 개선은 확인하지 못했습니다. 정확 전이 평가에서는 두 구조·두 prefix 가중치·모든 β의 macro gain MSE가 Mixed보다 높았습니다. 현재 비교에서는 Mixed를 유지하는 근거이며, 모든 학습 목적이나 PGM 구조의 가능성을 배제하는 결과는 아닙니다.')
    if 1 in betas and 'v1' in tr['weightings']:
        v='P1_independent';m=brier_name('v1',1);base='mixed_v1'
        if values[v,m,'gain_mse']<values[v,base,'gain_mse'] and exact_values[v,m,'gain_mse']>exact_values[v,base,'gain_mse']:
            conclusions.append(f"q 연결 제거 / v1 / β=1의 MC gain MSE 감소({values[v,m,'gain_mse']-values[v,base,'gain_mse']:+.8f})는 exact-event에서 {exact_values[v,m,'gain_mse']-exact_values[v,base,'gain_mse']:+.8f}로 바뀝니다. 이 작은 MC 이득을 native gain 예측의 개선으로 채택하지 않습니다. v1의 일부 ECE 개선은 기술통계로 남깁니다.")
    decomp_findings=[]
    if 1 in betas and 'uniform' in tr['weightings']:
        for v in VARIANTS:
            base=next(r for r in decomp_macro if r['variant']==v and r['training']=='mixed_uniform' and r['mode']=='exact_event')
            new=next(r for r in decomp_macro if r['variant']==v and r['training']=='brier_uniform_beta_1' and r['mode']=='exact_event')
            decomp_findings.append(f"{SHORT[v]} / uniform / β=1: exact Δrepair Brier={new['repair_brier']-base['repair_brier']:+.8f}, Δdamage Brier={new['damage_brier']-base['damage_brier']:+.8f}, Δerror cross={new['error_cross']-base['error_cross']:+.8f}. Test의 R/D Brier 자체도 개선되는지 먼저 확인해야 하며, 결과를 cross term 하나의 문제로 설명하지 않습니다.")
    qerr=max((r['max_error'] for r in tables['quadrature_checks'] if r['final_resolution'] and '__brier_' in r['candidate']),default=0)
    text=['# P1 Mixed + multi-step repair/damage Brier ablation',
        f"P1의 q 연결 유지/제거 모델에서 손실만 바꾸어 기존 Mixed 및 categorical 전이 NLL과 비교했습니다. {len(cells)}개 cell의 {split} 결과를 동일 가중 macro로 제시합니다. 모든 사전 고정 β를 보고하며 test에서 계수를 고르지 않습니다.",
        '## 핵심 결과',*conclusions,
        '## 손실과 통제',
        'B=mean(i,t,h)[((r−Yrepair)^2+(d−Ydamage)^2)/2], M=(J+λCw)/(1+λ), L=(M+βB)/(1+β)+ridge입니다. Repair와 damage의 두 binary Brier를 평균하며 stay의 제곱항, class balancing, gain-only 회귀항은 넣지 않습니다.',
        f"β={betas}를 모두 비교합니다. Brier는 전이 NLL과 손실 단위·gradient 크기가 달라 100까지 범위를 확장했습니다. 같은 숫자의 β가 NLL과 같은 학습 강도를 뜻하지는 않습니다. 기존 normalized loss 규칙과 raw ridge={tr['ridge']}는 유지합니다.",
        f"기존 Mixed의 λ와 파라미터 수 4/5개·입력 answer id/valid·prior/bounds·train cohort {sum(c['train_kept'] for c in cells)}개 model-question pair·L-BFGS·초기값 {tr['restart_seeds']}·σ 초기값 {tr['sigma_starts']}를 고정했습니다. Restart는 penalized train loss로 선택하고 β를 dev/test 점수로 선택하지 않습니다.",
        f"전이 학습 prefix={tr['transition_prefixes']}, horizon={tr['transition_horizons']}, t+h≤16의 쌍을 질문 안에서 동일 가중합니다. Uniform/v1은 posterior prefix 가중치의 별도 arm이고 B의 가중치는 같습니다.",
        '확률 계산은 이전 NLL 실험의 exact CRP path kernel과 q/Normal 적분을 그대로 재사용합니다. c,q,θ는 과거 답만 조건으로 주변화하며, 현재/미래 정답 여부는 target과 평가에만 씁니다. Native finite-horizon probability나 simulator를 변경하지 않았습니다. β=0은 기존 Mixed입니다.',
        '기존 Mixed·모든 전이 NLL β의 native/정확 전이 prediction 배열을 재사용하고 source bank/cohort/evaluation/code hash를 검증했습니다. 주 paired comparator는 사전에 고정한 NLL β=1입니다. 다른 NLL 계수도 CSV에 남깁니다.',
        '## Uniform: posterior와 다음 답 예측',
        table(['구조','학습','Unique NLL','Reference Brier','OTHER Brier','Next-answer NLL'],post_rows('uniform')),
        '## Uniform: 기존과 같은 paired MC',
        f"Seed {len(e['mc_seeds'])}개 × {e['simulations_per_seed']} continuations, 같은 질문/prefix/seed 난수·고정 plurality·first-appearance tie-break입니다. 같은 continuation에서 gain=repair−damage가 성립합니다. Bias는 예측−관측, pp 단위입니다. Gain ECE는 기존 고정 bin의 prefix/horizon별 calibration error를 cell 안에서 평균한 뒤 macro 평균한 기술통계입니다. ECE도 pp 단위이며 독립 표본의 calibration 검증으로 해석하지 않습니다.",
        table(['구조','학습','Repair Brier','Damage Brier','RD mean Brier','Gain MSE','Repair bias pp','Damage bias pp','Gain bias pp','Gain ECE pp'],gain_rows('uniform')),
        '## Exact-event 민감도',
        '같은 native 모델의 repair/damage/gain을 정확 path 합으로 계산해 MC 잡음을 제거한 보조 평가입니다. q/Normal 적분은 동일합니다. 이전 기준점의 exact 배열도 그대로 유지했습니다.',
        table(['구조','학습','Repair Brier','Damage Brier','RD mean Brier','Gain MSE','Repair bias pp','Damage bias pp','Gain bias pp'],gain_rows('uniform',True)),
        '## Posterior 개선 보존',
        '보존율=(Joint macro NLL−새 모델 macro NLL)/(Joint macro NLL−Mixed macro NLL)입니다. 100%는 기존 Mixed의 posterior 개선을 모두 보존한 경우이며 선택 조건으로 사용하지 않습니다. 분모가 양수가 아닌 cell의 개별 보존율은 정의하지 않습니다.',
        table(['구조','가중치','β','macro 보존율 %','Posterior<Joint cells','MC gain<Mixed cells','Exact gain<Mixed cells'],retained_rows),
        '## Gain 오차 분해',
        'eR=r−Yrepair, eD=d−Ydamage라 두면 Gain MSE=Repair Brier+Damage Brier−2 E[eR·eD]입니다. B는 앞 두 항의 평균이므로, 제한된 PGM 안에서 B의 감소만으로 gain MSE의 감소가 보장되지는 않습니다. gain_decomposition.csv와 gain_decomposition_macro.csv에 각 항과 항등식 잔차를 제공합니다.',
        *decomp_findings,
        '## 관찰된 변화',*findings,
        '## v1 가중치 민감도',
        table(['구조','학습','Unique NLL','Reference Brier','OTHER Brier','Next-answer NLL'],post_rows('v1')) if 'v1' in tr['weightings'] else 'v1은 이 실행에서 제외했습니다.',
        table(['구조','학습','Repair Brier','Damage Brier','RD mean Brier','Gain MSE','Repair bias pp','Damage bias pp','Gain bias pp','Gain ECE pp'],gain_rows('v1')) if 'v1' in tr['weightings'] else '',
        '## QA와 재현',
        f"새 Brier 후보의 최종 coarse/fine 최대 오차는 {qerr:.3g}입니다. Train J/C/B/data loss, dev posterior/Brier, 개별 dev posterior/next-answer/event 확률을 더 높은 Normal 해상도와 비교해 {cells[0]['study']['integration']['tolerance']} 이하인지 확인한 뒤 test를 평가합니다.",
        f"Frozen reference의 공통 summary 최대 차이는 {validation['frozen_summary_max_difference']:.3g}입니다. Posterior 정규화 오차 {validation['max_posterior_normalization_error']:.3g}, next-answer 정규화 오차 {validation['max_next_normalization_error']:.3g}, finite-draw gain 항등식 오차 {validation['max_gain_identity_error']:.3g}. validation.json에 기록했습니다.",
        'Brier 식과 기존 exact R/D 평가 점수의 일치, NLL과 같은 예측 확률, gold/suffix 격리, finite-difference gradient, β=0 Mixed 재현을 검증합니다. Kernel 자체의 경로 열거·horizon 8 독립 DP·native MC 검증도 유지합니다.',
        '## 그림',
        'Calibration 그림은 n≥30인 bin을 표시하며 전체 희소 bin은 CSV에 남깁니다. 상태 수를 독립 질문 수로 해석하지 않습니다.',
        '![Posterior uniform](figures/posterior_uniform.png)','![Repair damage uniform](figures/transitions_uniform.png)','![Gain uniform](figures/gain_uniform.png)',
        '## 범위와 한계',
        f"1. 질문 안에서 prefix/horizon 점수를 평균한 뒤 {e['bootstrap_replicates']:,}회 paired question bootstrap을 적용합니다. Macro는 cell 동일 가중 평균이며 pooled CI나 다중비교 유의성 주장을 하지 않습니다.",
        '2. 이미 탐색한 기존 xVerify exact-string bank이며 독립 holdout 검증은 아닙니다. 학습은 같은 unique-correct 질문, 주 미래 평가는 전체 질문이고 unique-only 민감도를 함께 저장합니다.',
        '3. λ 고정·현재 β grid·고정 horizon/plurality 규칙·normalized loss와 같은 raw ridge 아래의 결과입니다. NLL/Brier의 척도 차이와 강한 β에서의 상대 정규화 효과를 포함합니다. 결과로 β 또는 λ를 다시 고르지 않았습니다.',
        '4. 기존 P1의 q grid·bounded prior/σ·빈 답 생성 모델 부재가 남습니다. One-step transition-only 또는 gain-only 학습과의 비교는 아니며 실제 stopping threshold/토큰 절감도 최적화하지 않았습니다.',
        '## 실행 및 파일',
        'pgm p1-brier fit --workers 4','pgm p1-brier run --workers 4','pgm p1-brier report',
        'configs/p1_brier.json, protocol.json, selection/*.json에 고정 설정·코드 hash·모든 fit/restart/적분 이력·test 이전 lock을 저장합니다. artifacts/p1_brier/에는 후보와 큰 prediction 배열을 보관합니다.',
        'summary.csv / macro_summary.csv / dataset_summary.csv / paired_contrasts.csv / paired_macro.csv / exact_transition_summary.csv / exact_transition_macro.csv / exact_paired_contrasts.csv / exact_paired_macro.csv / posterior_retention.csv / cell_retention.csv / gain_decomposition.csv / gain_decomposition_macro.csv / *_calibration.csv / future_unique_summary.csv / training_restarts.csv / parameters.csv / quadrature_checks.csv / mc_seed_sensitivity.csv / mc_seed_contrasts.csv를 제공합니다.']
    markdown='\n\n'.join(x for x in text if x)+'\n';(out/'README.md').write_text(markdown)
    for w in tr['weightings']:figures(out,cells,w)
    style='body{font-family:system-ui,sans-serif;max-width:1450px;margin:40px auto;padding:0 24px;line-height:1.7;background:#fafbfc;color:#17212b}h1,h2{line-height:1.3;margin-top:2em}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:9px;border-bottom:1px solid #dce2e8}th{background:#eaf0f7}img{width:100%;margin:18px 0}'
    (out/'report.html').write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>P1 multi-step Brier ablation</title><style>"+style+'</style><body>'+markdown_html(markdown)+'</body></html>')
    return str(out/'report.html')
