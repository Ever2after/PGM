"""P1 review tables, paired contrasts, numerical diagnostics and figures."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from pgm_asc.diagnostics.report import csv_file, fmt, markdown_html, table
from pgm_asc.hierarchical.model import KEYS
from pgm_asc.objectives.report import make_figures
from pgm_asc.diagnostics import metrics
from pgm_asc.objectives.run import load_splits
from pgm_asc.utils import file_sha256

ORDER = tuple(KEYS)
LABELS = {"M1":"M1", "constant_theta":"Constant θ", "P1_independent":"P1 / q 연결 제거",
          "P1_linked":"P1 / q 연결 유지", "quadratic_theta":"Quadratic θ(q)"}
COLORS = dict(zip(ORDER, ['#3d6fa4','#777777','#d28136','#34846c','#9b6caf']))


def structure_contrasts(out, cells):
    """Direct paired ablations from saved predictions; no model fitting."""
    pairs = [('P1_linked', 'P1_independent'), ('P1_independent', 'constant_theta'),
             ('P1_linked', 'quadratic_theta')]
    rows = []
    for c in cells:
        data, bank_hash = load_splits(c['model'], c['dataset'], c['splits'])
        if bank_hash != c['bank_sha256'] or file_sha256(c['prediction_cache']['path']) != c['prediction_cache']['sha256']:
            raise ValueError('Saved prediction/bank provenance differs')
        e = c['study']['evaluation']
        with np.load(c['prediction_cache']['path'], allow_pickle=False) as z:
            for split, arrays in data.items():
                losses = {}
                for name in c['parameters']:
                    prefix = f'{split}.{name}.future.'
                    future = {k[len(prefix):]:z[k] for k in z.files if k.startswith(prefix)}
                    _, _, pl = metrics.posterior_losses(z[f'{split}.{name}.post'], arrays, e['prefixes'])
                    _, _, fl = metrics.future_scores(future, arrays, e['prefixes'], e['horizons'])
                    losses[name] = {**pl, **fl}
                for left, right in pairs:
                    if left not in losses or right not in losses:
                        continue
                    for metric, values in losses[left].items():
                        v = metrics.bootstrap_mean(values-losses[right][metric], e['bootstrap_replicates'], e['bootstrap_seed'])
                        ci = v['ci95'] or [None, None]
                        rows.append(dict(model=c['model'], dataset=c['dataset'], split=split,
                            comparison=f'{left} minus {right}', metric=metric, mean=v['mean'],
                            ci_low=ci[0], ci_high=ci[1], questions=v['questions']))
    csv_file(out/'structure_contrasts.csv', rows)
    macro = []
    for split in sorted({r['split'] for r in rows}):
        for comparison in sorted({r['comparison'] for r in rows}):
            for metric in sorted({r['metric'] for r in rows}):
                rr = [r for r in rows if r['split']==split and r['comparison']==comparison and r['metric']==metric and r['mean'] is not None]
                if rr:
                    macro.append(dict(split=split, comparison=comparison, metric=metric,
                        mean=float(np.mean([r['mean'] for r in rr])), cells=len(rr),
                        negative_cells=sum(r['mean']<0 for r in rr),
                        ci_below_zero_cells=sum(r['ci_high'] is not None and r['ci_high']<0 for r in rr),
                        ci_above_zero_cells=sum(r['ci_low'] is not None and r['ci_low']>0 for r in rr)))
    csv_file(out/'structure_macro.csv', macro)
    return macro


def report(out):
    out = Path(out)
    manifest = json.loads((out/'run.json').read_text())
    cells = [json.loads((out/p).read_text()) for p in manifest['cells']]
    if not cells or len({c['experiment_id'] for c in cells}) != 1:
        raise ValueError('Require a single-protocol nonempty run')
    fields = ('summary','future_unique_summary','posterior_by_prefix','future_by_horizon',
              'posterior_calibration','gain_calibration','transition_calibration','mc_seed_sensitivity',
              'paired_contrasts','parameters','training_restarts','quadrature_checks','baseline_reproduction')
    tables = {k:[] for k in fields}
    total_train = 0
    for c in cells:
        base = dict(model=c['model'],dataset=c['dataset'])
        fitted = json.loads((out/'selection'/f"{c['model']}_{c['dataset']}.json").read_text())
        total_train += fitted['train_questions']
        for name,fit in fitted['fits'].items():
            for r in fit['restarts']:
                tables['training_restarts'].append({**base,'method':name,
                    **{k:v for k,v in r.items() if k not in ('params','initial')},
                    'chosen_restart':r['restart_seed']==fit['best']['restart_seed']})
        for check in fitted['quadrature_checks']:
            tables['quadrature_checks'].append({**base,'method':check['variant'],
                'coarse_nodes':check['coarse_nodes'],'fine_nodes':check['fine_nodes'],
                **check['errors'],'max_error':check['max_error'],
                'selected_resolution':check['coarse_nodes']==fitted['normal_nodes']})
        if 'previous_M1_joint' in fitted:
            tables['baseline_reproduction'].append({**base,**fitted['previous_M1_joint']})
        for name,p in c['parameters'].items():
            tables['parameters'].append({**base,'method':name,'count':p['count'],**p['raw'],**p['readable'],
                'sigma_near_bound':p['readable']['sigma_theta'] > .95*c['study']['integration']['sigma_bound']})
        for split,s in c['splits'].items():
            for name,entry in s['models'].items():
                key={**base,'split':split,'method':name}
                for field in ('summary','future_unique_summary'):
                    for metric,v in entry[field].items():
                        ci=v['ci95'] or [None,None]
                        tables[field].append({**key,'metric':metric,'mean':v['mean'],
                            'ci_low':ci[0],'ci_high':ci[1],'questions':v['questions']})
                for field in ('posterior_by_prefix','future_by_horizon','posterior_calibration',
                              'gain_calibration','transition_calibration','mc_seed_sensitivity'):
                    tables[field] += [{**key,**r} for r in entry[field]]
            for name,diffs in s['paired_model_minus_M1'].items():
                for metric,v in diffs.items():
                    ci=v['ci95'] or [None,None]
                    tables['paired_contrasts'].append({**base,'split':split,'method':name,'comparison':'method minus M1',
                        'metric':metric,'mean':v['mean'],'ci_low':ci[0],'ci_high':ci[1],'questions':v['questions']})
    for name,rows in tables.items():csv_file(out/f'{name}.csv',rows)
    macro,dataset_rows=[],[]
    for split in sorted({r['split'] for r in tables['summary']}):
        for method in ORDER:
            for metric in sorted({r['metric'] for r in tables['summary']}):
                rr=[r for r in tables['summary'] if r['split']==split and r['method']==method and r['metric']==metric and r['mean'] is not None]
                if rr:
                    macro.append(dict(split=split,method=method,metric=metric,mean=float(np.mean([r['mean'] for r in rr])),cells=len(rr)))
                    for ds in sorted({r['dataset'] for r in rr}):
                        vs=[r['mean'] for r in rr if r['dataset']==ds]
                        dataset_rows.append(dict(split=split,dataset=ds,method=method,metric=metric,mean=float(np.mean(vs)),cells=len(vs)))
            bias=[abs(r['mean']) for r in tables['summary'] if r['split']==split and r['method']==method and r['metric']=='gain_bias']
            if bias:macro.append(dict(split=split,method=method,metric='abs_cell_gain_bias',mean=float(np.mean(bias)),cells=len(bias)))
    csv_file(out/'macro_summary.csv',macro)
    csv_file(out/'dataset_summary.csv',dataset_rows)
    split='test' if 'test' in cells[0]['splits'] else next(iter(cells[0]['splits']))
    contrasts = structure_contrasts(out, cells)
    contrast_rows = []
    for comparison in dict.fromkeys(r['comparison'] for r in contrasts):
        rr = {r['metric']: r for r in contrasts if r['split']==split and r['comparison']==comparison}
        contrast_rows.append([comparison]+[fmt(rr[k]['mean'],6) for k in
            ('unique_nll','next_answer_nll','repair_brier','damage_brier','gain_mse')])
    values={(r['method'],r['metric']):r['mean'] for r in macro if r['split']==split}
    methods=[name for name in ORDER if (name,'unique_nll') in values]
    def score(n,k,m=1.):return fmt(values[n,k]*m)
    post=[[LABELS[n],len(KEYS[n])]+[score(n,k) for k in ('unique_nll','reference_brier','other_brier','next_answer_nll')] for n in methods]
    future_events=[[LABELS[n]]+[score(n,k) for k in
        ('future_correct_brier','leader_change_brier','future_reference_correct_brier')] for n in methods]
    future=[[LABELS[n]]+[score(n,k) for k in ('repair_brier','damage_brier','gain_mse','gain_mse_minus_zero')]+[score(n,'gain_bias',100),score(n,'abs_cell_gain_bias',100)] for n in methods]
    transition=[]
    for name in methods:
        row=[LABELS[name]]
        for event in ('repair','damage'):
            for k in ('prediction','observed'):
                row.append(fmt(100*np.mean([np.mean([r[k] for r in c['splits'][split]['models'][name]['future_by_horizon'] if r['event']==event]) for c in cells]),2))
        transition.append(row)
    details=[]
    for c in cells:
        m=c['splits'][split]['models']
        details.append([f"{c['model']} / {c['dataset']}"]+
            [fmt(m[n]['summary']['unique_nll']['mean']) for n in methods]+
            [fmt(100*m[n]['summary']['gain_bias']['mean'],2) for n in methods])
    sigmas=[]
    for c in cells:
        p=c['parameters']
        sigmas.append([f"{c['model']} / {c['dataset']}",c['normal_nodes']]+
            [fmt(p[n]['readable']['sigma_theta'],3) for n in ('P1_independent','P1_linked') if n in p]+
            [fmt(p[n]['readable']['theta_q_slope'],3) for n in ('M1','P1_linked') if n in p])
    changes=[]
    for name in methods:
        if name=='M1':continue
        improved=[]
        for metric in ('unique_nll','next_answer_nll','repair_brier','damage_brier','gain_mse'):
            diffs=[c['splits'][split]['models'][name]['summary'][metric]['mean']-c['splits'][split]['models']['M1']['summary'][metric]['mean'] for c in cells]
            improved.append(f"{metric} {values[name,metric]-values['M1',metric]:+.6f} ({sum(d<0 for d in diffs)}/{len(cells)} cells 개선)")
        changes.append(f"{LABELS[name]} − M1: "+'; '.join(improved)+'.')
    lessons=[]
    if 'P1_linked' in methods:
        pct={k:100*(values['P1_linked',k]/values['M1',k]-1) for k in
             ('unique_nll','next_answer_nll','gain_mse')}
        lessons.append(f"P1 linked의 M1 대비 상대 변화는 posterior NLL {pct['unique_nll']:+.2f}%, "
            f"next-answer NLL {pct['next_answer_nll']:+.2f}%, gain MSE {pct['gain_mse']:+.2f}%입니다. "
            "Posterior/답 분포의 개선 크기와 stopping에 필요한 gain 예측의 개선 크기를 구분해야 합니다.")
        lessons.append(f"Repair bias는 M1 {100*values['M1','repair_bias']:+.3f}pp → "
            f"P1 linked {100*values['P1_linked','repair_bias']:+.3f}pp, damage bias는 "
            f"{100*values['M1','damage_bias']:+.3f}pp → {100*values['P1_linked','damage_bias']:+.3f}pp입니다. "
            "Gain bias=repair bias−damage bias이므로 damage 보정이 좋아져도 repair 과대예측이 남으면 gain bias가 커질 수 있습니다. "
            "P1만으로 기대 이득 calibration 문제가 해결됐다고 판단하지 않습니다.")
    if 'P1_linked' in methods and 'P1_independent' in methods:
        lessons.append(f"q 연결을 유지할 때 independent 대비 Δposterior NLL="
            f"{values['P1_linked','unique_nll']-values['P1_independent','unique_nll']:+.6f}, "
            f"Δnext-answer NLL={values['P1_linked','next_answer_nll']-values['P1_independent','next_answer_nll']:+.6f}, "
            f"Δgain MSE={values['P1_linked','gain_mse']-values['P1_independent','gain_mse']:+.6f}입니다. "
            "추가 연결의 가치는 posterior·미래 답·전이 목적에 따라 다르므로 한 metric으로 유지/제거를 확정하지 않습니다.")
    if 'P1_linked' in methods and 'quadratic_theta' in methods:
        count=sum(c['splits'][split]['models']['P1_linked']['summary']['unique_nll']['mean'] <
                  c['splits'][split]['models']['quadratic_theta']['summary']['unique_nll']['mean'] for c in cells)
        lessons.append(f"같은 5-parameter quadratic 대조군보다 linked P1의 posterior NLL이 {count}/{len(cells)}개 cell에서 낮습니다. "
            "이 범위에서는 결정론적 θ(q)에 2차항만 추가하는 설명보다 문제별 잔여 이질성을 모델링하는 설명이 더 지지됩니다.")
    maxerr=max((r['max_error'] for r in tables['quadrature_checks'] if r['selected_resolution']),default=0)
    maxgap=max((abs(r['penalized_train_gap']) for r in tables['baseline_reproduction']),default=0)
    e=cells[0]['study']['evaluation'];tr=cells[0]['study']['training'];integ=cells[0]['study']['integration']
    text=['# P1 계층 CRP 구조 ablation','',
        f"기존 16-response bank의 {len(cells)}개 cell에서 동일한 joint 학습으로 구조만 비교했습니다. 주 결과는 {split}의 cell 동일 가중 macro 평균입니다.",'',
        '## 구조와 통제','',
        'q~Beta(a,b), u~Normal(0,1), log θ=clip(t0+t1(2q−1),−8,8)+σu입니다. u는 문제마다 하나이며 전체 prefix와 미래 응답에서 공유됩니다. θ가 작을수록 오답이 소수 문자열에 집중됩니다. 계층 변동은 cell 안의 문제 간 θ 차이를 나타냅니다.',
        table(['방법','q 연결','문제별 random θ','자유 파라미터'],[
            ['M1','linear','없음',4],['Constant θ','없음','없음',3],
            ['P1 independent','없음','있음',4],['P1 linked','linear','있음',5],
            ['Quadratic θ(q)','quadratic','없음',5]]),'',
        'q 연결 제거는 prior에서 q와 θ를 독립으로 둔다는 뜻입니다. 답의 관측 및 latent c를 통한 posterior 의존성은 남습니다. Quadratic 대조군은 linked P1과 파라미터 수를 맞춰, 단순한 θ(q) 유연성 증가와 random effect를 구분합니다.',
        f"모든 방법은 같은 unique-correct train 질문 {sum(c['train_kept'] for c in cells)}/{total_train}개 model-question pair, 같은 answer id/valid 입력을 사용합니다. 정답 라벨은 train target과 점수 계산에만 사용합니다.",
        f"학습 목적 J=−log p(D16,c*)/16 + {tr['ridge']}×Σraw². 이전 M1 Joint와 같은 정규화·ridge·L-BFGS·공통 초기값 {tr['restart_seeds']}입니다. 파라미터 수가 다른 방법도 raw 하나당 같은 ridge를 씁니다. 각 구조는 penalized train loss가 최소인 restart를 선택하며 dev/test 성능으로 구조를 선택하지 않습니다.",
        f"추가 σ는 smooth signed loading {integ['sigma_bound']}×tanh(raw/{integ['sigma_bound']})로 나타내며 부호는 대칭 정규분포에서 동일합니다. 보고하는 σ는 절댓값입니다. σ=0에서 random effect가 정확히 사라집니다. 0에서 gradient가 0인 variance component이므로 양의 σ 초기값 {tr['sigma_starts']}를 사용했습니다.",'',
        '## Posterior·다음 답 예측','',
        '모든 loss는 낮을수록 좋습니다. Unique NLL은 bank에 정답 판정 문자열이 최대 하나인 질문에서, OTHER를 포함한 정답 위치 posterior를 평가합니다. Reference/OTHER/미래 점수는 전체 질문을 사용합니다. 다음 답의 categorical 분포는 적분으로 계산하며 MC를 쓰지 않습니다. 추가 미래 사건별 NLL/Brier는 future_by_horizon.csv에 있습니다.','',
        table(['방법','Params','Unique NLL','Reference Brier','OTHER Brier','Next-answer NLL'],post),'',
        table(['방법','Future-correct Brier','Leader-change Brier','Future-reference Brier'],future_events),'',
        '## Repair·damage·gain calibration','',
        f"prefix={e['prefixes']}, horizon={e['horizons']}, t+h≤16의 상태를 질문 안에서 먼저 평균했습니다. 예측은 seed {len(e['mc_seeds'])}개 × {e['simulations_per_seed']} continuations입니다. 모든 구조에 동일한 질문/prefix/seed별 난수와 inverse-CDF 순서를 사용합니다. c,q,θ를 공동으로 한 번 뽑고 continuation 동안 고정합니다.",
        '동일 continuation에서 gain=미래 정오−현재 정오=repair−damage입니다. 실제 결과도 bank의 같은 현재/미래 plurality 비교입니다. 동점은 먼저 등장한 valid 답을 선택합니다. Gain bias는 예측−관측이며 pp 단위입니다. Gain MSE−zero<0이면 0-gain 예측보다 좋습니다.','',
        table(['방법','Repair Brier','Damage Brier','Gain MSE','MSE−zero','Gain bias pp','평균 절대 cell bias pp'],future),'',
        table(['방법','Repair 예측 %','실제 %','Damage 예측 %','실제 %'],transition),'',
        '## 해석','',*[x for p in lessons for x in (p,'')],
        '## 관찰된 변화','',*[x for p in changes for x in (p,'')],
        'Random effect 유무의 주 비교는 M1↔P1 linked, Constant↔P1 independent입니다. q 연결의 주 비교는 P1 independent↔P1 linked입니다. 한 metric의 개선이 다른 층의 개선을 보장하지 않습니다. Gain의 signed bias만으로는 repair와 damage 오차의 상쇄를 구분할 수 없습니다.','',
        '아래 직접 비교도 같은 질문과 continuation의 paired contrast입니다. 음수는 왼쪽 모델의 loss가 더 낮음을 뜻합니다. Cell별 질문 bootstrap CI는 structure_contrasts.csv, 개선 cell 수는 structure_macro.csv에 있습니다.','',
        table(['비교','ΔUnique NLL','ΔNext NLL','ΔRepair Brier','ΔDamage Brier','ΔGain MSE'],contrast_rows),'',
        '![Posterior by prefix](figures/posterior.png)','',
        '## Cell별 결과','',
        table(['Cell']+[f'{LABELS[n]} NLL' for n in methods]+[f'{LABELS[n]} bias pp' for n in methods],details),'',
        table(['Cell','Normal nodes','σ independent','σ linked','M1 slope','P1 linked slope'],sigmas),'',
        '## 적분·최적화 검증','',
        f"q grid는 기존 M1의 80개 logit [−7,7]을 유지합니다. u는 정규분포 Gauss-Hermite로 적분합니다. 해상도 후보 {integ['normal_nodes']}에서 인접 해상도의 train joint NLL, dev unique NLL, dev 개별 posterior/next-answer 최대 절대 차이가 모두 {integ['tolerance']} 이하일 때 채택합니다. 그렇지 않으면 양쪽 P1을 함께 높은 해상도로 재학습합니다. 선택된 해상도의 최대 오차는 {maxerr:.3g}입니다. Test 점수는 수치 해상도 선택에 사용하지 않았습니다.",
        f"이전 controlled M1 Joint와 동일 train ID/hash를 확인했으며 penalized train objective의 최대 차이는 {maxgap:.3g}입니다. Nested σ=0 likelihood·posterior·gradient, label 격리, 확률 정규화, 공동 latent 추출, shared simulation parity는 자동 검증합니다.",
        'CRP rising factorial은 log(θ+r)의 누적 합으로 계산해 큰 θ에서 lgamma 차분 소실을 피합니다. Normal random effect 자체는 clip하지 않습니다. θ(q)의 기존 base clip만 유지합니다. σ 상한 근접 여부, 모든 restart/gradient, 적분 오차는 parameters.csv, training_restarts.csv, quadrature_checks.csv에 기록합니다.','',
        '![Repair and damage](figures/transitions.png)','',
        '![Gain calibration](figures/gain.png)','',
        '## 범위와 한계','',
        '1. 기존 xVerify candidate 라벨과 exact-string 답 후보를 재사용합니다. 의미가 같은 여러 정답 문자열은 주 Unique NLL에서 제외하며 future_unique_summary.csv에 미래 민감도를 따로 제공합니다. 새 judge 판정은 수행하지 않았습니다.',
        '2. Paired contrast CI는 질문 단위 bootstrap입니다. Prefix/horizon과 MC seed를 독립 질문으로 세지 않습니다. Macro에 pooled CI나 다중비교 유의성 주장은 없습니다. 전체 CI는 paired_contrasts.csv에 있습니다.',
        '3. MC repair/damage는 Brier와 reliability를 우선합니다. 유한 draw에서 0/1 확률이 생길 수 있어 MC NLL은 clipping=1e−12, zero_probability_events와 함께 해석해야 합니다. MC seed sensitivity와 MC noise 보정 gain MSE도 제공합니다.',
        '4. q grid 절단·σ 상한·특정 Normal prior·단일 joint 목적 내 결과입니다. 적분 검사 통과는 구조 또는 prior가 맞다는 의미가 아닙니다. 학습 bank·judge·구조 선택의 불확실성은 질문 bootstrap에 포함되지 않습니다.',
        '5. 기존 M1과 동일하게 빈 답을 생성하는 observation model이 없습니다. 실제 invalid 다음 답에 0 확률을 줍니다. Stopping threshold나 토큰 절감 성능은 이번 실험에서 최적화하지 않았습니다.',
        '6. 이미 탐색한 test bank이므로 새로운 구조의 독립적인 최종 검증이 아닙니다. 구조·loss를 확정한 뒤 별도 질문 holdout이 필요합니다.','',
        '## 재현','',
        'pgm hierarchical fit --workers 2','pgm hierarchical run --workers 2','pgm hierarchical report','',
        'configs/hierarchical.json과 protocol.json은 설정·코드 hash를, selection/*.json은 test를 읽기 전 fit lock을 담습니다. 큰 posterior/forecast 배열은 artifacts/hierarchical_crp/에 저장합니다. 모든 표는 이 폴더의 CSV와 cells/*.json으로 제공합니다.']
    markdown='\n\n'.join(x for x in text if x)+'\n'
    (out/'README.md').write_text(markdown)
    make_figures(out,cells,{k:COLORS[k] for k in methods}, {**LABELS,**{k:k for k in methods}})
    style='body{font-family:system-ui,sans-serif;max-width:1440px;margin:40px auto;padding:0 24px;line-height:1.7;background:#fafbfc;color:#17212b}h1,h2{line-height:1.3;margin-top:2em}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:9px;border-bottom:1px solid #dce2e8}th{background:#eaf0f7}img{width:100%;margin:18px 0}'
    (out/'report.html').write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>P1 계층 CRP</title><style>"+style+'</style><body>'+markdown_html(markdown)+'</body></html>')
    return str(out/'report.html')
