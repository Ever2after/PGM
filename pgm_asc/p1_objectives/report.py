"""P1 objective scorecards, calibration plots and paired question comparisons."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from pgm_asc import config
from pgm_asc.diagnostics.report import csv_file,fmt,markdown_html,table
from pgm_asc.p1_objectives.run import baseline_root,method_key

VARIANTS=('P1_independent','P1_linked')
SHORT={'P1_independent':'q 연결 제거','P1_linked':'q 연결 유지'}
METHODS=('joint','prefix_uniform','mixed_uniform','prefix_v1','mixed_v1')
LABELS={'joint':'Joint','prefix_uniform':'Prefix / uniform','mixed_uniform':'Mixed / uniform',
        'prefix_v1':'Prefix / v1','mixed_v1':'Mixed / v1'}


def figures(out,cells,weighting):
    cache=config.ARTIFACTS/'p1_objectives'/'matplotlib'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache));os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    folder=out/'figures';folder.mkdir(exist_ok=True)
    split='test' if 'test' in cells[0]['splits'] else next(iter(cells[0]['splits']))
    datasets=sorted({c['dataset'] for c in cells})
    methods=('joint',f'prefix_{weighting}',f'mixed_{weighting}')
    colors=dict(zip(methods,('#3d6fa4','#d28136','#34846c')))
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
                                    marker='o',color=colors[method],label=LABELS[method])
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
                                ax.plot(x,y,color=colors[method],label=LABELS[method])
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
        axes[0,-1].legend(fontsize=8)
        fig.tight_layout();fig.savefig(folder/f'{chart}_{weighting}.png',dpi=150,bbox_inches='tight');plt.close(fig)


def report(out):
    out=Path(out)
    manifest=json.loads((out/'run.json').read_text())
    cells=[json.loads((out/p).read_text()) for p in manifest['cells']]
    if not cells or len({c['experiment_id'] for c in cells})!=1:raise ValueError('Require one nonempty protocol')
    fields=('summary','future_unique_summary','posterior_by_prefix','posterior_calibration','future_by_horizon',
            'gain_calibration','transition_calibration','mc_seed_sensitivity','paired_contrasts',
            'training_restarts','quadrature_checks','selection','parameters')
    tables={k:[] for k in fields}
    validation=dict(cells=len(cells),joint_arrays_identical=True,joint_summary_max_difference=0.,
        max_posterior_normalization_error=0.,max_next_normalization_error=0.,max_gain_identity_error=0.,
        fit_locks_precede_evaluation=True,lambda_ranking_stable=True)
    for c in cells:
        base=dict(model=c['model'],dataset=c['dataset'])
        fitted=json.loads((out/'selection'/f"{c['model']}_{c['dataset']}.json").read_text())
        if c['selection_locked_at']>c['evaluation_started_at']:raise ValueError('Fit was not locked before evaluation')
        if any(x['choice_changed'] for x in fitted['selection_checks']):raise ValueError('Numerically unstable lambda selection')
        for name,f in fitted['fits'].items():
            for hi,history in enumerate(f.get('resolution_history') or [f]):
                for restart in history['restarts']:
                    tables['training_restarts'].append({**base,'candidate':name,'variant':f['variant'],
                        **f['objective'],**{k:v for k,v in restart.items() if k not in ('params','initial')},
                        'final_resolution':hi==max(len(f.get('resolution_history',[]))-1,0),
                        'chosen_restart':restart['restart_seed']==history['best']['restart_seed'],
                        'frozen_previous_joint':f.get('frozen_previous_joint',False)})
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
                    if method=='joint':
                        for k in z.files:
                            if k.startswith(f'{split}.{name}.'):
                                original=k.replace(f'{split}.{name}.',f'{split}.{variant}.',1)
                                if not np.array_equal(z[k],old[original],equal_nan=True):raise ValueError('Frozen Joint array changed')
                        for metric,v in entry['summary'].items():
                            previous=prior['splits'][split]['models'][variant]['summary'][metric]['mean']
                            if v['mean'] is not None and previous is not None:
                                validation['joint_summary_max_difference']=max(validation['joint_summary_max_difference'],abs(v['mean']-previous))
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
    methods=[m for m in METHODS if ('P1_linked',m,'unique_nll') in values]
    def score(v,m,k,multiplier=1.,digits=4):return fmt(values[v,m,k]*multiplier,digits)
    def post_rows(ms):return [[SHORT[v],LABELS[m]]+[score(v,m,k) for k in
        ('unique_nll','reference_brier','other_brier','next_answer_nll')] for v in VARIANTS for m in ms]
    def gain_rows(ms):return [[SHORT[v],LABELS[m]]+[score(v,m,k,digits=5) for k in
        ('repair_brier','damage_brier','gain_mse')]+[score(v,m,k,100,3) for k in
        ('repair_bias','damage_bias','gain_bias','abs_cell_gain_bias')] for v in VARIANTS for m in ms]
    primary=[m for m in ('joint','prefix_uniform','mixed_uniform') if m in methods]
    selection_rows=[]
    for c in cells:
        row=[f"{c['model']} / {c['dataset']}"]
        for v in VARIANTS:
            for w in c['study']['training']['weightings']:
                f=c['parameters'][method_key(v,f'mixed_{w}')]
                row.append(f['objective']['mixing'])
        selection_rows.append(row)
    findings=[]
    lessons=[]
    for v in VARIANTS:
        for m in primary:
            if m=='joint':continue
            changes={k:values[v,m,k]-values[v,'joint',k] for k in
                ('unique_nll','next_answer_nll','repair_brier','damage_brier','gain_mse')}
            wins={k:sum(c['splits'][split]['models'][method_key(v,m)]['summary'][k]['mean']<
                c['splits'][split]['models'][method_key(v,'joint')]['summary'][k]['mean'] for c in cells) for k in changes}
            findings.append(f"{SHORT[v]} / {LABELS[m]} − Joint: "+'; '.join(
                f'{k} {changes[k]:+.6f} ({wins[k]}/{len(cells)} cells 개선)' for k in changes)+'.')
        if 'prefix_uniform' in methods:
            lessons.append(f"{SHORT[v]}의 uniform Prefix는 Joint 대비 posterior NLL "
                f"{100*(values[v,'prefix_uniform','unique_nll']/values[v,'joint','unique_nll']-1):+.2f}%, "
                f"gain MSE {100*(values[v,'prefix_uniform','gain_mse']/values[v,'joint','gain_mse']-1):+.2f}%인 반면, "
                f"next-answer NLL은 {100*(values[v,'prefix_uniform','next_answer_nll']/values[v,'joint','next_answer_nll']-1):+.2f}%입니다. "
                "Posterior·gain의 보정과 전체 미래 답 분포의 proper score 사이에 trade-off가 남습니다.")
            lessons.append(f"{SHORT[v]}의 repair bias는 {100*values[v,'joint','repair_bias']:+.3f} → "
                f"{100*values[v,'prefix_uniform','repair_bias']:+.3f}pp로 바뀌고, damage bias는 "
                f"{100*values[v,'joint','damage_bias']:+.3f} → {100*values[v,'prefix_uniform','damage_bias']:+.3f}pp입니다. "
                "Gain bias 감소에는 repair 변화와 damage 변화가 함께 반영됩니다. Damage proper score와 calibration이 모두 개선됐다고 해석하지 않습니다.")
        if 'mixed_uniform' in methods and 'mixed_v1' in methods:
            lessons.append(f"{SHORT[v]}의 Mixed에서 v1 가중치는 uniform 대비 next-answer NLL을 "
                f"{values[v,'mixed_uniform','next_answer_nll']:.4f} → {values[v,'mixed_v1','next_answer_nll']:.4f}로 바꾸지만, "
                f"Joint의 {values[v,'joint','next_answer_nll']:.4f}까지 회복하지는 않습니다. "
                f"Gain MSE는 {values[v,'mixed_uniform','gain_mse']:.5f} → {values[v,'mixed_v1','gain_mse']:.5f}입니다. "
                "가중치 선택도 posterior, 생성 예측, gain 중 어떤 목적을 중시하는지에 따라 달라집니다.")
    dsrows=[]
    for ds in sorted({c['dataset'] for c in cells}):
        for v in VARIANTS:
            for m in primary:
                rr={r['metric']:r['mean'] for r in dataset_rows if r['split']==split and r['dataset']==ds and r['variant']==v and r['training']==m}
                dsrows.append([ds,SHORT[v],LABELS[m]]+[fmt(rr[k],5) for k in ('unique_nll','next_answer_nll','gain_mse')]+[fmt(100*rr['gain_bias'],3)])
    e=cells[0]['study']['evaluation'];tr=cells[0]['study']['training']
    qerr=max((r['max_error'] for r in tables['quadrature_checks'] if r['final_resolution']),default=0)
    future_rows=[[SHORT[v],LABELS[m]]+[score(v,m,k) for k in
        ('future_correct_brier','leader_change_brier','future_reference_correct_brier')] for v in VARIANTS for m in primary]
    text=['# P1 학습 목적 ablation',
        f"두 P1 구조의 입력과 파라미터 수를 유지하고 Joint / prefix posterior NLL / mixed 학습을 비교했습니다. {len(cells)}개 cell의 {split} 결과이며 cell을 동일하게 가중한 macro 평균입니다.",
        '## 고정한 조건과 dev 선택',
        'q 연결 제거형은 4개, 유지형은 5개 파라미터입니다. q~Beta(a,b), log θ=clip(t0+t1(2q−1),−8,8)+σu, u~Normal(0,1)이며 제거형은 t1=0입니다. u는 문제마다 한 번 정해지고 전체 prefix/continuation에서 공유됩니다. 입력은 answer id와 valid뿐입니다.',
        'J=−mean log p(D16,c*)/16. Cw=질문별 Σ(t=1..16) wt[−log p(c*t|Dt)]의 평균입니다. Joint=J, Prefix=Cw, Mixed=(J+λCw)/(1+λ)이며 모두 같은 raw-parameter ridge를 더합니다. 정답이 아직 나타나지 않은 prefix는 OTHER가 target입니다.',
        f"모든 학습은 기존 Joint와 정확히 같은 unique-correct train 질문 {sum(c['train_kept'] for c in cells)}개 model-question pair를 사용합니다. Bank 안에 정답 판정 문자열이 최대 하나인 질문이며, 정답이 한 번도 나오지 않은 질문도 유지합니다. 정답 라벨은 train target과 평가에만 사용합니다.",
        f"초기값 seed {tr['restart_seeds']}, σ 초기값 {tr['sigma_starts']}, L-BFGS, ridge={tr['ridge']}를 유지합니다. 각 목적은 penalized train loss로 restart를 선택합니다. Mixed λ 후보 {tr['lambda_grid']} 중 dev unique posterior NLL이 가장 낮은 값을 선택합니다. Dev 점수는 고정 평가 prefix의 동일 가중 평균입니다. λ 선택에 미래/gain/test 점수는 쓰지 않습니다.",
        'Uniform은 16개 prefix에 같은 가중치를 줍니다. v1은 1–2/3–4/5–8/9–16 band에 총 0.4/0.2/0.2/0.2를 주며 별도 민감도로 비교합니다. 모든 λ 후보와 restart를 저장합니다.',
        'Joint 기준점은 이전 P1 실험의 파라미터와 posterior/forecast 배열을 그대로 재사용했습니다. Bank hash·train ID hash·평가 설정·수치 구현 hash를 검증하고, 현재 점수가 이전 점수와 같은지 확인합니다. 기존 결과 파일은 덮어쓰지 않습니다.',
        '## Posterior와 미래 예측: uniform',
        '모든 loss는 낮을수록 좋습니다. Unique NLL은 unique-correct 질문, reference/OTHER/미래 점수는 전체 질문을 사용합니다. Next-answer 분포는 적분으로 계산하며 MC를 쓰지 않습니다.',
        table(['구조','학습','Unique NLL','Reference Brier','OTHER Brier','Next-answer NLL'],post_rows(primary)),
        table(['구조','학습','Future-correct Brier','Leader-change Brier','Future-reference Brier'],future_rows),
        '## Repair·damage·gain calibration: uniform',
        f"동일 질문/prefix/seed 난수를 모든 모델에 공유합니다. Prefix={e['prefixes']}, horizon={e['horizons']}, t+h≤16이며 seed {len(e['mc_seeds'])}개 × {e['simulations_per_seed']} draws입니다. Latent c,q,θ를 공동으로 한 번 뽑고 미래 응답 동안 고정합니다. 같은 continuation에서 gain=repair−damage가 유한 draw에서도 성립합니다.",
        'Reference는 valid 답의 plurality이며 동점은 먼저 등장한 답을 택합니다. Bias는 예측−관측, 단위는 pp입니다. 전체 signed gain bias와 평균 절대 cell bias를 함께 제시해 cell 간 오차 상쇄를 구분합니다.',
        table(['구조','학습','Repair Brier','Damage Brier','Gain MSE','Repair bias pp','Damage bias pp','Gain bias pp','평균 절대 cell bias pp'],gain_rows(primary)),
        '## 해석',*lessons,
        '## Joint 대비 변화',*findings,
        'Gain MSE와 calibration bias는 다른 성질입니다. Posterior NLL만 좋아지거나 signed gain bias가 상쇄되는 경우를 전체 개선으로 해석하지 않습니다. Paired contrast CSV에는 각 구조의 Joint 비교, 같은 목적의 q 연결 유지−제거, Mixed−Prefix를 모두 담았습니다.',
        '## 데이터셋별 결과',
        table(['Dataset','구조','학습','Unique NLL','Next NLL','Gain MSE','Gain bias pp'],dsrows),
        '## v1 가중치 민감도',
        table(['구조','학습','Unique NLL','Reference Brier','OTHER Brier','Next-answer NLL'],post_rows(methods)),
        table(['구조','학습','Repair Brier','Damage Brier','Gain MSE','Repair bias pp','Damage bias pp','Gain bias pp','평균 절대 cell bias pp'],gain_rows(methods)),
        '## Cell별 mixed λ 선택',
        table(['Cell']+[f'{SHORT[v]} / {w}' for v in VARIANTS for w in tr['weightings']],selection_rows),
        '## 적분과 재현 검증',
        f"원래 q grid 80개 logit [−7,7]과 σ 상한 {cells[0]['study']['integration']['sigma_bound']}을 유지합니다. 각 후보는 기존 Joint의 Normal 적분 해상도에서 시작해 더 높은 해상도의 train J/Cw/data loss, dev unique NLL, 개별 dev posterior/next-answer 확률과 비교합니다. 최대 오차 허용값은 {cells[0]['study']['integration']['tolerance']}이고 이번 선택 해상도의 최대 오차는 {qerr:.3g}입니다. 실패하면 해당 목적을 재학습하며 test 전에 검사를 끝냅니다. Mixed λ의 순위도 더 높은 해상도에서 유지되어야 합니다.",
        f"Joint 배열은 이전 cache와 동일합니다. Joint summary의 최대 차이는 {validation['joint_summary_max_difference']:.3g}, posterior 정규화 오차는 {validation['max_posterior_normalization_error']:.3g}, next-answer 정규화 오차는 {validation['max_next_normalization_error']:.3g}, gain 항등식 오차는 {validation['max_gain_identity_error']:.3g}입니다. validation.json에 저장했습니다.",
        '## Calibration 곡선',
        '같은 데이터셋의 cell/prefix/horizon에서 같은 bin을 합친 descriptive reliability입니다. 그림에는 n≥30인 bin을 표시하며 점 크기는 상태 수입니다. 반복 상태들은 독립 표본이 아닙니다. 희소한 끝 구간을 포함한 전체 bin은 CSV의 n과 함께 확인할 수 있습니다.',
        '![Posterior uniform](figures/posterior_uniform.png)','![Repair damage uniform](figures/transitions_uniform.png)',
        '![Gain uniform](figures/gain_uniform.png)',
        '## 범위와 한계',
        '1. 기존 xVerify candidate 라벨과 exact-string 후보를 재사용합니다. 다중 정답 문자열은 주 unique posterior 점수와 분리하며 future_unique_summary.csv에 미래 점수의 unique-correct 민감도를 제공합니다.',
        f"2. 질문 안에서 prefix/horizon 점수를 먼저 평균한 뒤 {e['bootstrap_replicates']:,}회 paired question bootstrap을 적용합니다. Cell macro에는 pooled CI나 다중비교 유의성 주장을 붙이지 않습니다. 모델/목적 선택의 불확실성과 judge 오차는 이 CI에 포함되지 않습니다.",
        '3. Mixed λ는 dev posterior 목적에 맞춘 선택입니다. 미래/gain에 최적인 λ는 아닐 수 있습니다. 테스트 결과로 목적·가중치·λ를 다시 고르지 않았습니다. 이미 탐색한 bank이므로 독립 holdout의 최종 검증은 아닙니다.',
        '4. MC repair/damage는 Brier와 reliability를 우선합니다. MC NLL은 0/1 확률과 clipping=1e−12에 민감하므로 zero_probability_events와 함께 봅니다. seed별 결과와 MC 보정 gain MSE도 저장합니다.',
        '5. 기존 P1과 같은 q grid 절단, bounded prior/σ, 빈 답을 생성하지 않는 한계가 남습니다. Stopping threshold와 토큰 절감 성능은 이번 실험에서 선택하지 않았습니다.',
        '## 재현 및 파일',
        'pgm p1-objective fit --workers 4','pgm p1-objective run --workers 4','pgm p1-objective report',
        'configs/p1_objectives.json, protocol.json, selection/*.json에 설정·코드 hash·test 이전 선택 lock을 기록합니다. artifacts/p1_objectives/에는 후보별 적분 이력·모든 restart·큰 prediction 배열을 보관합니다.',
        'summary.csv, macro_summary.csv, dataset_summary.csv, paired_contrasts.csv, paired_macro.csv, posterior_by_prefix.csv, future_by_horizon.csv, *_calibration.csv, future_unique_summary.csv, selection.csv, training_restarts.csv, parameters.csv, quadrature_checks.csv, mc_seed_sensitivity.csv, mc_seed_contrasts.csv를 제공합니다.']
    markdown='\n\n'.join(text)+'\n'
    (out/'README.md').write_text(markdown)
    for weighting in tr['weightings']:figures(out,cells,weighting)
    style='body{font-family:system-ui,sans-serif;max-width:1450px;margin:40px auto;padding:0 24px;line-height:1.7;background:#fafbfc;color:#17212b}h1,h2{line-height:1.3;margin-top:2em}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:9px;border-bottom:1px solid #dce2e8}th{background:#eaf0f7}img{width:100%;margin:18px 0}'
    (out/'report.html').write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>P1 objective ablation</title><style>"+style+'</style><body>'+markdown_html(markdown)+'</body></html>')
    return str(out/'report.html')
