"""P1 predictive-loss ablation: frozen references, all gamma arms, paired comparisons."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from pgm_asc import config
from pgm_asc.diagnostics.report import csv_file,fmt,markdown_html,table
from pgm_asc.p1_predictive.run import baseline_root,method_key,predictive_name
from pgm_asc.p1_predictive import CONFIG

VARIANTS=('P1_independent','P1_linked')
SHORT={'P1_independent':'q 연결 제거','P1_linked':'q 연결 유지'}
METHODS=('joint','prefix_uniform','mixed_uniform','prefix_v1','mixed_v1')+tuple(predictive_name(w,g) for w in CONFIG['training']['weightings'] for g in CONFIG['training']['predictive_weights'])
LABELS={'joint':'Joint','prefix_uniform':'Prefix / uniform','mixed_uniform':'Mixed / uniform',
        'prefix_v1':'Prefix / v1','mixed_v1':'Mixed / v1'}


def plot_label(method):
    if method.startswith('predictive_'):
        return 'Mixed + next, gamma='+method.split('_gamma_')[1].replace('p','.')
    return LABELS[method]


def figures(out,cells,weighting):
    cache=config.ARTIFACTS/'p1_predictive'/'matplotlib'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache));os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    folder=out/'figures';folder.mkdir(exist_ok=True)
    split='test' if 'test' in cells[0]['splits'] else next(iter(cells[0]['splits']))
    datasets=sorted({c['dataset'] for c in cells})
    methods=[m for m in ('joint',f'prefix_{weighting}',f'mixed_{weighting}')+tuple(predictive_name(weighting,g) for g in CONFIG['training']['predictive_weights']) if method_key(VARIANTS[0],m) in cells[0]['parameters']]
    colors=dict(zip(methods,('#888888','#d0a65b','#3d6fa4','#4f9d69','#9b6caf','#d65f5f')))
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


def tradeoff_figure(out,values,weighting,gammas):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    colors={'P1_independent':'#3d6fa4','P1_linked':'#34846c'}
    methods=[f'mixed_{weighting}']+[predictive_name(weighting,g) for g in gammas]
    for ax,metric in zip(axes,('unique_nll','gain_mse')):
        for vi,v in enumerate(VARIANTS):
            x=[values[v,m,'next_answer_nll'] for m in methods]
            y=[values[v,m,metric] for m in methods]
            ax.plot(x,y,'o-',color=colors[v],label=v.removeprefix('P1_'))
            for i,(xx,yy,g) in enumerate(zip(x,y,[0.]+gammas)):
                ax.annotate(f'{g:g}',(xx,yy),xytext=(5,7 if (i+vi)%2 else -13),textcoords='offset points',fontsize=8,color=colors[v])
            for m,marker in (('joint','s'),(f'prefix_{weighting}','D')):
                ax.scatter(values[v,m,'next_answer_nll'],values[v,m,metric],marker=marker,s=48,
                    color=colors[v],edgecolors='white',zorder=4)
        ax.set(xlabel='Next-answer NLL (lower is better)',ylabel='Posterior NLL' if metric=='unique_nll' else 'Gain MSE')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5));ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    handles += [Line2D([],[],color='gray',marker='s',lw=0,label='Joint'),Line2D([],[],color='gray',marker='D',lw=0,label='Prefix')]
    axes[0].legend(handles=handles,fontsize=8)
    fig.suptitle(f'{weighting}: Mixed + next loss; labels show gamma (0 = Mixed)')
    fig.tight_layout();fig.savefig(out/'figures'/f'tradeoff_{weighting}.png',dpi=160,bbox_inches='tight');plt.close(fig)


def report(out):
    out=Path(out)
    manifest=json.loads((out/'run.json').read_text())
    cells=[json.loads((out/p).read_text()) for p in manifest['cells']]
    if not cells or len({c['experiment_id'] for c in cells})!=1:raise ValueError('Require one nonempty protocol')
    fields=('summary','future_unique_summary','posterior_by_prefix','posterior_calibration','future_by_horizon',
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
                    for field in ('summary','future_unique_summary'):
                        for metric,v in entry[field].items():
                            ci=v['ci95'] or [None,None]
                            tables[field].append({**key,'metric':metric,'mean':v['mean'],'ci_low':ci[0],
                                'ci_high':ci[1],'questions':v['questions']})
                    for field in ('posterior_by_prefix','posterior_calibration','future_by_horizon','gain_calibration',
                                  'transition_calibration','mc_seed_sensitivity'):
                        tables[field]+=[{**key,**r} for r in entry[field]]
                    post=z[f'{split}.{name}.post'];nxt=z[f'{split}.{name}.future.next_answer']
                    validation['max_posterior_normalization_error']=max(validation['max_posterior_normalization_error'],float(np.max(np.abs(post.sum(-1)-1))))
                    validation['max_next_normalization_error']=max(validation['max_next_normalization_error'],float(np.nanmax(np.abs(nxt.sum(-1)-1))))
                    gain=z[f'{split}.{name}.future.gain'];repair=z[f'{split}.{name}.future.repair'];damage=z[f'{split}.{name}.future.damage']
                    validation['max_gain_identity_error']=max(validation['max_gain_identity_error'],float(np.nanmax(np.abs(gain-repair+damage))))
                    if not method.startswith('predictive_'):
                        for k in z.files:
                            if k.startswith(f'{split}.{name}.'):
                                original=k
                                if not np.array_equal(z[k],old[original],equal_nan=True):raise ValueError('Frozen reference array changed')
                        for metric,v in entry['summary'].items():
                            previous=prior['splits'][split]['models'][name]['summary'][metric]['mean']
                            if v['mean'] is not None and previous is not None:
                                validation['frozen_summary_max_difference']=max(validation['frozen_summary_max_difference'],abs(v['mean']-previous))
                for comparison,diffs in s['paired_contrasts'].items():
                    for metric,v in diffs.items():
                        ci=v['ci95'] or [None,None]
                        tables['paired_contrasts'].append({**base,'split':split,'comparison':comparison,
                            'metric':metric,'mean':v['mean'],'ci_low':ci[0],'ci_high':ci[1],'questions':v['questions']})
    for name,rows in tables.items():csv_file(out/f'{name}.csv',rows)
    macro,dataset_rows=[],[]
    for split in sorted({r['split'] for r in tables['summary']}):
        for variant in VARIANTS:
            for method in METHODS:
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
        if r['training']=='joint':continue
        reference=next(x for x in sensitivity if x['model']==r['model'] and x['dataset']==r['dataset']
            and x['split']==r['split'] and x['variant']==r['variant'] and x['training']=='joint' and x['seed']==r['seed'])
        for metric in ('gain_mse','repair_brier','damage_brier'):
            mc_contrasts.append({k:r[k] for k in ('model','dataset','split','variant','training','seed')} |
                dict(metric=metric,difference=r[metric]-reference[metric]))
    csv_file(out/'mc_seed_contrasts.csv',mc_contrasts)
    from pgm_asc.utils import write_json
    write_json(out/'validation.json',validation)
    split='test' if 'test' in cells[0]['splits'] else next(iter(cells[0]['splits']))
    values={(r['variant'],r['training'],r['metric']):r['mean'] for r in macro if r['split']==split}
    tr=cells[0]['study']['training'];e=cells[0]['study']['evaluation']
    gammas=tr['predictive_weights']
    def methods_for(w):return ['joint',f'prefix_{w}',f'mixed_{w}']+[predictive_name(w,g) for g in gammas]
    def score(v,m,k,scale=1.,digits=4):return fmt(values[v,m,k]*scale,digits)
    def post_rows(w):return [[SHORT[v],plot_label(m)]+[score(v,m,k) for k in
        ('unique_nll','reference_brier','other_brier','next_answer_nll')] for v in VARIANTS for m in methods_for(w)]
    def gain_rows(w):return [[SHORT[v],plot_label(m)]+[score(v,m,k,digits=5) for k in
        ('repair_brier','damage_brier','gain_mse')]+[score(v,m,k,100,3) for k in
        ('repair_bias','damage_bias','gain_bias','abs_cell_gain_bias')] for v in VARIANTS for m in methods_for(w)]
    def future_rows(w):return [[SHORT[v],plot_label(m)]+[score(v,m,k) for k in
        ('next_answer_brier','future_correct_brier','leader_change_brier','future_reference_correct_brier')]
        for v in VARIANTS for m in methods_for(w)]
    retention=[]
    for c in cells:
        for split_name,s in c['splits'].items():
            for v in VARIANTS:
                def val(m,k):return s['models'][method_key(v,m)]['summary'][k]['mean']
                for w in tr['weightings']:
                    base=f'mixed_{w}';improvement=val('joint','unique_nll')-val(base,'unique_nll')
                    for g in gammas:
                        m=predictive_name(w,g)
                        row=dict(model=c['model'],dataset=c['dataset'],split=split_name,variant=v,weighting=w,gamma=g)
                        for reference in ('joint',base):
                            tag='joint' if reference=='joint' else 'mixed'
                            for metric in ('unique_nll','next_answer_nll','gain_mse','repair_brier','damage_brier'):
                                row[f'{metric}_minus_{tag}']=val(m,metric)-val(reference,metric)
                            row[f'abs_gain_bias_minus_{tag}']=abs(val(m,'gain_bias'))-abs(val(reference,'gain_bias'))
                            row[f'three_losses_below_{tag}']=all(val(m,k)<val(reference,k) for k in ('unique_nll','next_answer_nll','gain_mse'))
                        row['posterior_gain_retained_fraction']=(val('joint','unique_nll')-val(m,'unique_nll'))/improvement if improvement>0 else None
                        retention.append(row)
    csv_file(out/'posterior_retention.csv',retention)
    retained_rows=[];findings=[];lessons=[]
    for v in VARIANTS:
        for w in tr['weightings']:
            baseline=f'mixed_{w}';den=values[v,'joint','unique_nll']-values[v,baseline,'unique_nll']
            for g in gammas:
                m=predictive_name(w,g)
                fraction=(values[v,'joint','unique_nll']-values[v,m,'unique_nll'])/den if den>0 else None
                rr=[r for r in retention if r['split']==split and r['variant']==v and r['weighting']==w and r['gamma']==g]
                retained_rows.append([SHORT[v],w,g,fmt(100*fraction,1) if fraction is not None else 'N/A',
                    sum(r['unique_nll_minus_joint']<0 for r in rr),sum(r['next_answer_nll_minus_joint']<0 for r in rr),
                    sum(r['gain_mse_minus_joint']<0 for r in rr),sum(r['three_losses_below_joint'] for r in rr),
                    sum(r['three_losses_below_mixed'] for r in rr)])
                findings.append(f"{SHORT[v]} / {w} / γ={g:g}: Mixed 대비 Δposterior NLL "
                    f"{values[v,m,'unique_nll']-values[v,baseline,'unique_nll']:+.6f}, Δnext-answer NLL "
                    f"{values[v,m,'next_answer_nll']-values[v,baseline,'next_answer_nll']:+.6f}, Δgain MSE "
                    f"{values[v,m,'gain_mse']-values[v,baseline,'gain_mse']:+.6f}. "
                    f"Joint 대비 세 loss가 모두 낮은 cell은 {sum(r['three_losses_below_joint'] for r in rr)}/{len(rr)}개입니다.")
        if 1. in gammas:
            m=predictive_name('uniform',1.);base='mixed_uniform'
            den=values[v,'joint','unique_nll']-values[v,base,'unique_nll']
            retained=(values[v,'joint','unique_nll']-values[v,m,'unique_nll'])/den if den>0 else np.nan
            lessons.append(f"{SHORT[v]} / uniform γ=1은 Mixed의 macro posterior 개선 중 {100*retained:.1f}%를 보존하면서 "
                f"next-answer NLL을 {values[v,base,'next_answer_nll']:.4f} → {values[v,m,'next_answer_nll']:.4f}로 바꿉니다. "
                f"그러나 gain MSE는 {values[v,base,'gain_mse']:.5f} → {values[v,m,'gain_mse']:.5f}, "
                f"평균 절대 cell gain bias는 {100*values[v,base,'abs_cell_gain_bias']:.3f} → {100*values[v,m,'abs_cell_gain_bias']:.3f}pp입니다. "
                "Posterior 보존·다음 답 예측 회복과 Mixed 대비 gain calibration의 추가 개선을 구분해야 합니다.")
            lessons.append(f"{SHORT[v]} / uniform γ=1의 repair bias는 Mixed의 "
                f"{100*values[v,base,'repair_bias']:+.3f} → {100*values[v,m,'repair_bias']:+.3f}pp, "
                f"damage bias는 {100*values[v,base,'damage_bias']:+.3f} → {100*values[v,m,'damage_bias']:+.3f}pp입니다. "
                "다음 답 예측이 회복되는 동안 repair 과대예측도 다시 증가했습니다.")
        if all(values[v,predictive_name(w,g),'gain_mse']>=values[v,f'mixed_{w}','gain_mse']
               for w in tr['weightings'] for g in gammas):
            lessons.append(f"{SHORT[v]}에서는 이번 두 가중치·모든 γ의 macro gain MSE가 각각의 Mixed보다 낮아지지 않았습니다. "
                "희소한 cell별 개선이나 작은 차이를 전체 calibration 해결로 해석하지 않습니다. 질문 단위 CI와 MC 보정/seed 민감도를 함께 봅니다.")
    qerr=max((r['max_error'] for r in tables['quadrature_checks'] if r['final_resolution']),default=0)
    sections=['# P1 Mixed + next-answer predictive NLL ablation',
        f"동일한 두 P1 그래프에서 기존 Mixed에 next-answer predictive NLL을 추가했습니다. {len(cells)}개 cell의 {split} 결과를 동일 가중 macro로 보고합니다. γ를 test에서 선택하지 않고 모든 사전 고정 arm을 제시합니다.",
        '## 학습 목적과 통제',
        'J=−mean log p(D16,c*)/16, Cw=16개 prefix의 posterior NLL 가중 평균, M=(J+λCw)/(1+λ)입니다. 추가 손실 N=mean(i,t=1..15)[−log p(answer(i,t+1)|D(i,t))]로 두고 L=(M+γN)/(1+γ)+ridge를 학습합니다. γ=0은 기존 Mixed입니다. λ가 달라도 γ가 정규화된 Mixed에 대한 상대 가중치를 뜻하도록 정의했습니다.',
        f"λ는 이전 dev-selected Mixed 값을 cell·구조·prefix 가중치별로 그대로 고정했습니다. γ 후보는 {gammas}이며 모두 별도 arm으로 학습·평가합니다. Posterior 보존 허용폭이나 미래/gain 점수로 γ를 고르는 절차를 추가하지 않았습니다.",
        'N은 latent 정답 c, 성공률 q, 문제별 θ를 적분한 기존 P1의 native 다음 답 분포입니다. Gold c나 posterior 평균을 대입하지 않고 예측 moment를 직접 적분합니다. Valid 전이는 연속 prefix의 marginal evidence 비율과도 일치하는지 교차 검증했습니다. 기존 답/NEW/INVALID categorical target을 사용하며 미래 답은 target에만 쓰고 조건부 입력은 과거 prefix만 씁니다.',
        'N의 학습은 모든 t=1..15, 다음 답 평가는 기존 t=1/2/4/8입니다. 미래 라벨과 관련 없는 단일 답 예측 항이며 정답 라벨을 읽지 않습니다. Invalid 답의 native 확률은 0이고 기존 평가와 같은 1e−12 clipping을 적용하므로 해당 항은 gradient가 없는 상수입니다. 별도의 빈 답 observation model은 추가하지 않았습니다.',
        f"파라미터 수는 q 연결 제거형 4개, 유지형 5개입니다. 입력·prior·question-shared θ·q grid·train 질문 {sum(c['train_kept'] for c in cells)}개 model-question pair·초기값 {tr['restart_seeds']}·σ 초기값 {tr['sigma_starts']}·ridge={tr['ridge']}·L-BFGS를 유지했습니다. Restart는 penalized train loss로 선택했습니다.",
        '기존 Joint/Prefix/Mixed의 fit과 모든 prediction 배열은 이전 실험에서 그대로 가져왔습니다. Source bank/cohort/evaluation/code hash와 prediction SHA·질문 순서를 검증했습니다. Test 평가 전에 새 fit lock을 저장했습니다.',
        '## Uniform: posterior와 다음 답',
        '모든 loss는 낮을수록 좋습니다. Unique posterior 점수는 bank에 정답 판정 문자열이 최대 하나인 질문에서 계산하며, reference/OTHER/미래 점수는 전체 질문을 사용합니다.',
        table(['구조','학습','Unique NLL','Reference Brier','OTHER Brier','Next-answer NLL'],post_rows('uniform')),
        table(['구조','학습','Next Brier','Future-correct Brier','Leader-change Brier','Future-reference Brier'],future_rows('uniform')),
        '## Uniform: repair·damage·gain calibration',
        f"평가 prefix={e['prefixes']}, horizon={e['horizons']}, t+h≤16. 동일한 질문/prefix/seed별 난수, seed {len(e['mc_seeds'])}개 × {e['simulations_per_seed']} continuations를 사용했습니다. c,q,θ를 한 번 공동 추출하고 미래 응답 동안 공유합니다. Gain=repair−damage가 finite-draw에서도 성립합니다. Bias는 예측−관측, pp 단위입니다.",
        table(['구조','학습','Repair Brier','Damage Brier','Gain MSE','Repair bias pp','Damage bias pp','Gain bias pp','평균 절대 cell bias pp'],gain_rows('uniform')),
        '## 해석',*lessons,
        '## Posterior 개선 보존과 동시 개선',
        '보존율은 macro NLL의 (Joint−새 모델)/(Joint−기존 Mixed)입니다. 100%는 Mixed의 posterior 개선을 모두 보존한 경우이며, 음수는 Joint보다도 나빠진 경우입니다. Cell별 분모가 양수가 아닌 경우의 보존율은 정의하지 않았습니다. 어떠한 비율도 γ 선택 조건으로 사용하지 않았습니다.',
        '세 loss 동시 개선은 unique posterior NLL·next-answer NLL·gain MSE가 해당 reference보다 모두 낮은 cell 수입니다. Signed gain bias의 상쇄를 성공 조건으로 쓰지 않습니다. paired_contrasts.csv에 각 비교의 질문 단위 신뢰구간을 제공합니다.',
        table(['구조','가중치','γ','macro 보존율 %','Posterior<Joint cells','Next<Joint cells','Gain MSE<Joint cells','세 loss<Joint cells','세 loss<Mixed cells'],retained_rows),
        '## 관찰된 변화',*findings,
        '## v1 가중치 민감도',
        table(['구조','학습','Unique NLL','Reference Brier','OTHER Brier','Next-answer NLL'],post_rows('v1')) if 'v1' in tr['weightings'] else 'v1 가중치는 이 실행에 포함되지 않았습니다.',
        table(['구조','학습','Repair Brier','Damage Brier','Gain MSE','Repair bias pp','Damage bias pp','Gain bias pp','평균 절대 cell bias pp'],gain_rows('v1')) if 'v1' in tr['weightings'] else '',
        '## 수치 검증',
        f"Native P1 예측기와 next-answer target 확률 및 gradient를 검증했습니다. 관측 suffix와 gold 라벨을 바꿔 prefix 예측이 오염되지 않는지 확인했습니다. γ=0에서 이전 Mixed 손실과 gradient를 재현합니다. 각 fit은 더 높은 Normal 적분 해상도로 train J/C/N/data loss, dev posterior/next NLL 및 개별 dev posterior/next 확률을 확인했습니다. 최대 허용 오차는 {cells[0]['study']['integration']['tolerance']}, 이번 선택 해상도의 최대 오차는 {qerr:.3g}입니다.",
        f"모든 frozen reference 배열은 동일하고, summary 최대 차이는 {validation['frozen_summary_max_difference']:.3g}입니다. Posterior 정규화 오차 {validation['max_posterior_normalization_error']:.3g}, next 정규화 오차 {validation['max_next_normalization_error']:.3g}, gain 항등식 오차 {validation['max_gain_identity_error']:.3g}. validation.json에 기록했습니다.",
        '## 그림',
        'Trade-off 그림의 원은 Mixed(γ=0)와 세 predictive 가중치를 잇습니다. 왼쪽 아래일수록 두 loss가 낮습니다. 정사각형은 Joint, 마름모는 Prefix이며, 그림은 cell 동일 가중 macro를 요약합니다.',
        '![Predictive trade-off](figures/tradeoff_uniform.png)',
        'Calibration 그림은 n≥30 bin을 표시합니다. 점 크기는 상태 수이며 반복 prefix/horizon은 독립 표본이 아닙니다. 전체 희소 bin은 CSV에 남깁니다.',
        '![Posterior uniform](figures/posterior_uniform.png)',
        '![Repair damage uniform](figures/transitions_uniform.png)',
        '![Gain uniform](figures/gain_uniform.png)',
        '## 범위와 한계',
        f"1. 질문 내부 prefix/horizon 평균을 만든 뒤 {e['bootstrap_replicates']:,}회 paired question bootstrap을 적용합니다. Macro는 cell 동일 가중 평균이며 pooled CI나 다중비교 유의성 주장을 하지 않습니다. MC seed와 반복 상태를 독립 질문으로 세지 않습니다.",
        '2. 모든 γ를 사전에 고정했고 test 결과를 이용해 γ 또는 λ를 다시 선택하지 않았습니다. 이미 탐색한 기존 bank의 ablation이며 독립 holdout 검증은 아닙니다. λ를 γ와 함께 재탐색하는 실험도 아닙니다.',
        '3. 기존 xVerify 라벨, exact-string 후보, q grid 절단, bounded prior/σ, 빈 답 생성 모델 부재의 한계가 남습니다. Unique-correct train에 맞춘 결과이며 전체 질문의 미래 점수와 unique-only 민감도를 함께 제공합니다.',
        '4. One-step predictive NLL의 개선은 multi-step repair/damage와 stopping 개선을 보장하지 않습니다. MC NLL은 clipping과 zero-probability event에 민감해 Brier/reliability를 우선하며, MC 보정 gain MSE와 seed별 contrast도 제공합니다. 실제 stopping threshold와 토큰 절감은 이번 실험에서 최적화하지 않았습니다.',
        '## 재현',
        'pgm p1-predictive fit --workers 4','pgm p1-predictive run --workers 4','pgm p1-predictive report',
        'configs/p1_predictive.json, protocol.json, selection/*.json에 사전 설정·고정 λ·모든 γ arm·code hash·fit lock을 기록합니다. artifacts/p1_predictive/에 모든 candidate/restart/적분 이력과 큰 prediction 배열을 저장합니다.',
        'summary.csv, macro_summary.csv, dataset_summary.csv, paired_contrasts.csv, paired_macro.csv, posterior_retention.csv, posterior_by_prefix.csv, future_by_horizon.csv, *_calibration.csv, future_unique_summary.csv, training_restarts.csv, parameters.csv, quadrature_checks.csv, selection.csv, mc_seed_sensitivity.csv, mc_seed_contrasts.csv를 제공합니다.']
    markdown='\n\n'.join(x for x in sections if x)+'\n';(out/'README.md').write_text(markdown)
    for w in tr['weightings']:
        figures(out,cells,w)
        tradeoff_figure(out,values,w,gammas)
    style='body{font-family:system-ui,sans-serif;max-width:1450px;margin:40px auto;padding:0 24px;line-height:1.7;background:#fafbfc;color:#17212b}h1,h2{line-height:1.3;margin-top:2em}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:9px;border-bottom:1px solid #dce2e8}th{background:#eaf0f7}img{width:100%;margin:18px 0}'
    (out/'report.html').write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>P1 predictive ablation</title><style>"+style+'</style><body>'+markdown_html(markdown)+'</body></html>')
    return str(out/'report.html')
