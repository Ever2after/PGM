"""A focused P1 stopping report from the unchanged v2 method evaluation."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from pgm_asc import config
from pgm_asc.diagnostics.report import csv_file,markdown_html,table
from pgm_asc.p1_stopping.run import definitions,out_dir,code_hashes
from pgm_asc.utils import read_json,write_json
from pgm_asc.v2 import CONFIG


CORE=['p1_independent_joint','p1_linked_joint','p1_independent_mixed','p1_linked_mixed']
BASE=['sc','M1','asc','esc','reasc_offline','reasc_online']
LABEL={'sc':'SC@k','M1':'M1 VOI','asc':'ASC','esc':'ESC','reasc_offline':'ReASC off','reasc_online':'ReASC on'}


def report():
    root=out_dir();protocol=read_json(root/'protocol.json');defs=definitions(protocol['study'])
    if protocol['v2']!=CONFIG or protocol['code_sha256']!=code_hashes():raise ValueError('Stopping protocol/code mismatch')
    LABEL.update({m:d['label'] for m,d in defs.items()})
    cells=[];operating=[];frontier=[];contrasts=[];qa=[]
    max_baseline_error=0.
    for model in CONFIG['models']:
        for dataset in CONFIG['datasets']:
            name=f'{model}_{dataset}';base=read_json(config.RESULTS/'v2/cells'/f'{name}.json')
            lock=read_json(root/'cells'/f'{name}.json')
            if lock['experiment_id']!=protocol['experiment_id']:raise ValueError('Mixed stopping protocols')
            if lock['locked_at']>lock['evaluation_observations_loaded_at']:raise ValueError('Parameters not locked before evaluation')
            row=dict(model=model,dataset=dataset,sc16=base['sc16']['test'],methods={m:base for m in BASE})
            for m in defs:
                result=read_json(config.RESULTS/'v2/methods'/m/f'{name}.json')
                row['methods'][m]=result
                if result['sc16']!=base['sc16'] or result['frontier_budgets']!=base['frontier_budgets']:
                    raise ValueError('Baseline reference/budget changed')
                for split in ('dev','test'):
                    for b in ('sc','M1'):
                        if len(base['curves'][split][b])!=len(result['curves'][split][b]):raise ValueError('Baseline curve length changed')
                        for original,copy in zip(base['curves'][split][b],result['curves'][split][b]):
                            if original['threshold']!=copy['threshold']:raise ValueError('Baseline threshold changed')
                            for k in ('accuracy','mean_tokens','mean_samples'):
                                max_baseline_error=max(max_baseline_error,abs(original[k]-copy[k]))
                for rule,c in result['contrasts'].items():
                    d=c[m]
                    contrasts.append(dict(model=model,dataset=dataset,method=m,rule=rule,
                        accuracy_difference_p1_minus_m1=d['accuracy_difference']*-1,
                        accuracy_ci_low=-d['accuracy_difference_ci95'][1],accuracy_ci_high=-d['accuracy_difference_ci95'][0],
                        token_ratio_p1_over_m1=1/d['token_ratio'],token_ratio_ci_low=1/d['token_ratio_ci95'][1],
                        token_ratio_ci_high=1/d['token_ratio_ci95'][0],mcnemar_exact_p=d['mcnemar_exact_p']))
            cells.append(row)
            for m,r in row['methods'].items():
                frontier.append(dict(model=model,dataset=dataset,method=m,frontier_score=r['frontier_score'][m]))
                for rule,sel in r['selected'].items():
                    if m not in sel:continue
                    p=sel[m]['test']
                    operating.append(dict(model=model,dataset=dataset,method=m,rule=rule,threshold=str(sel[m]['threshold']),
                        accuracy=p['accuracy'],mean_tokens=p['mean_tokens'],mean_samples=p['mean_samples'],
                        token_share_of_sc16=p['mean_tokens']/base['sc16']['test']['mean_tokens']))
            for head,h in lock['heads'].items():
                qa.append(dict(model=model,dataset=dataset,head=head,normal_nodes=h['normal_nodes'],
                    source_selected=h['source_selected'],mixing=h['objective'].get('mixing'),
                    max_log_evidence_error=h['integration_checks'][-1]['max_log_evidence_error'],
                    max_lookup_posterior_error=max(v for k,v in lock['count_lookup_posterior_errors'].items() if k.endswith(head))))
    if max_baseline_error>1e-12:raise ValueError('Rescored M1/SC differ from frozen baseline')
    for name,rows in [('operating_points',operating),('frontier',frontier),('paired_vs_m1',contrasts),('numerical_checks',qa)]:
        csv_file(root/f'{name}.csv',rows)
    macro=[]
    for m in BASE+list(defs):
        f=float(np.mean([r['frontier_score'] for r in frontier if r['method']==m]))
        for rule in sorted({r['rule'] for r in operating}):
            rr=[r for r in operating if r['method']==m and r['rule']==rule]
            if rr:macro.append(dict(method=m,rule=rule,cells=len(rr),frontier_score=f,
                **{k:float(np.mean([r[k] for r in rr])) for k in ('accuracy','mean_tokens','mean_samples','token_share_of_sc16')}))
    csv_file(root/'macro_summary.csv',macro)
    values={(r['method'],r['rule']):r for r in macro}
    def macro_rows(methods,rule='cf_accuracy'):
        return [[LABEL[m],f"{100*values[m,rule]['frontier_score']:.3f}",f"{100*values[m,rule]['accuracy']:.3f}",
            f"{100*values[m,rule]['token_share_of_sc16']:.2f}",f"{values[m,rule]['mean_samples']:.3f}"] for m in methods]
    def point_rows(methods,rule):
        return [[c['model'],c['dataset'],f"{100*c['sc16']['accuracy']:.1f}"]+[
            f"{100*c['methods'][m]['selected'][rule][m]['test']['accuracy']:.1f} / {c['methods'][m]['selected'][rule][m]['test']['mean_tokens']:.0f}" for m in methods] for c in cells]
    findings=[]
    for m in defs:
        f=values[m,'cf_accuracy'];b=values['M1','cf_accuracy']
        wins=sum(c['methods'][m]['frontier_score'][m]>c['methods']['M1']['frontier_score']['M1'] for c in cells)
        matched=[r for r in contrasts if r['method']==m and r['rule']=='cf_accuracy']
        findings.append(f"{LABEL[m]}: M1 대비 macro frontier {100*(f['frontier_score']-b['frontier_score']):+.3f} pp "
            f"({wins}/{len(cells)} cell 증가), cross-fitted accuracy {100*(f['accuracy']-b['accuracy']):+.3f} pp, "
            f"SC16 token share {100*(f['token_share_of_sc16']-b['token_share_of_sc16']):+.2f} pp. "
            f"개별 paired accuracy CI가 0보다 높은/낮은 cell은 "
            f"{sum(r['accuracy_ci_low']>0 for r in matched)}/{sum(r['accuracy_ci_high']<0 for r in matched)}개입니다.")
    max_quad=max(r['max_log_evidence_error'] for r in qa);max_lookup=max(r['max_lookup_posterior_error'] for r in qa)
    validation=dict(cells=len(cells),heads=len(qa),methods=len(defs),frozen_baseline_max_difference=max_baseline_error,
        max_count_lookup_posterior_error=max_lookup,max_count_space_log_evidence_error=max_quad,
        fit_locks_precede_evaluation=True,v2_score_families_unchanged=True)
    write_json(root/'validation.json',validation)
    main=BASE+CORE
    allp1=CORE+['p1_independent_mixed_v1','p1_linked_mixed_v1','p1_independent_mixed_confidence','p1_linked_mixed_confidence']
    header=['학습·기준','Frontier %','CF accuracy %','CF tokens / SC16 %','CF samples']
    conclusions=[]
    voi_methods=[m for m,d in defs.items() if d['stop']=='voi']
    if all(values[m,'cf_accuracy']['frontier_score']<values['M1','cf_accuracy']['frontier_score'] for m in voi_methods):
        gaps=[100*(values[m,'cf_accuracy']['frontier_score']-values['M1','cf_accuracy']['frontier_score']) for m in voi_methods]
        conclusions.append(f"P1 VOI {len(voi_methods)}개 설정 모두 기존 M1보다 macro frontier가 {abs(max(gaps)):.3f}–{abs(min(gaps)):.3f} pp 낮았습니다. 현재 결과에서 P1이 M1의 accuracy–token trade-off를 개선했다고 보기 어렵습니다. Frontier는 test oracle 점추정이며 유의성 검정값은 아닙니다.")
    for v in ('independent','linked'):
        m=f'p1_{v}_mixed';c=m+'_confidence';j=f'p1_{v}_joint'
        a=values[m,'cf_accuracy'];b=values['M1','cf_accuracy'];conf=values[c,'cf_accuracy'];joint=values[j,'cf_accuracy']
        conclusions.append(f"q {v} Mixed-uniform VOI는 M1 대비 CF accuracy {100*(a['accuracy']-b['accuracy']):+.3f} pp, "
            f"SC16 token share {100*(a['token_share_of_sc16']-b['token_share_of_sc16']):+.2f} pp입니다. "
            f"같은 구조의 Joint 대비 frontier는 {100*(a['frontier_score']-joint['frontier_score']):+.3f} pp로, "
            '이전 posterior NLL 개선이 stopping 성능의 일관된 개선으로 이어지지 않았습니다.')
        conclusions.append(f"같은 {v} Mixed posterior의 confidence control은 VOI보다 CF token share가 "
            f"{100*(conf['token_share_of_sc16']-a['token_share_of_sc16']):+.2f} pp 높고 frontier는 "
            f"{100*(conf['frontier_score']-a['frontier_score']):+.3f} pp 낮습니다. CF accuracy는 "
            f"{100*(conf['accuracy']-a['accuracy']):+.3f} pp 높으므로 이 operating point를 동일 정확도 비교로 해석하지 않습니다.")
    text=['# P1 adaptive self-consistency stopping 결과',
        '이전 P1 실험은 고정 prefix에서 PGM 자체를 평가했습니다. 이번에는 기존 파라미터를 재학습하지 않고 VOI stopping을 붙여 실제 accuracy–output-token 결과를 평가했습니다. Confidence stopping은 같은 Mixed-uniform posterior의 대조군입니다.',
        '## 핵심 결과',*conclusions,
        '## 비교와 프로토콜',
        'q 연결 제거/유지 각각 Joint·Mixed uniform·Mixed v1의 6개 고정 head에 VOI를 적용하고, Mixed uniform의 confidence 대조군 2개를 더했습니다. 기존 dev-selected 학습 λ와 파라미터 수 4/5개를 유지하며 stopping 결과로 학습 head를 고르지 않았습니다.',
        'README와 docs/PROTOCOL.md의 Python plugin 경로를 사용했습니다. `pgm v2 method --plugin examples/p1_stopping.py`가 기존 `score_families`를 그대로 호출하고, 기존 M1·SC와 함께 새 method JSON을 생성합니다. M1/ASC/ESC/ReASC와 bank·응답 순서·정답 판정·출력 토큰 비용 정의가 같습니다.',
        f"최소 {CONFIG['stopping']['minimum']}개·최대 {CONFIG['stopping']['cap']}개 응답, 기존 λ {len(CONFIG['lambdas'])}개와 γ {len(CONFIG['gammas'])}개를 그대로 사용했습니다. VOI는 기존처럼 {CONFIG['voi']['simulations']}회 MC, seed={CONFIG['voi']['seed']}, horizon={CONFIG['voi']['horizons']} 및 남은 cap입니다. VOI=max(0,max_h (E[A(t+h)]−A(t))/h)이며 추가 응답당 이득입니다. 출력 토큰당 VOI로 바꾸지 않았습니다.",
        '현재·미래 답은 v2의 plurality와 posterior tie-break이고, 새 미래 답을 얻은 뒤 P1 posterior를 다시 계산한 confidence를 평균합니다. c,q,θ는 continuation마다 한 번 뽑아 유지합니다. P1 arm 사이에는 같은 난수를 사용합니다.',
        f"Primary operating point는 test 2-fold cross-fitting(seed={CONFIG['crossfit']['seed']})으로 선택한 SC16−1 pp matched accuracy입니다. Dev 선택과 20/30/50% 예산도 보고합니다. Frontier는 SC@2–SC16 사이 60개 예산에서의 test oracle curve 요약이며 독립 검증 성능으로 해석하지 않습니다.",
        f"Paired bootstrap은 질문 단위 {CONFIG['bootstrap']['replicates']:,}회, seed={CONFIG['bootstrap']['seed']}와 exact McNemar입니다. JSON의 공식 contrast는 M1−P1이고, paired_vs_m1.csv는 읽기 쉽게 P1−M1 및 P1/M1 token ratio로 뒤집었습니다. Macro는 9개 cell 동일 가중 기술통계로 pooled 유의성을 주장하지 않습니다.",
        '## Macro: cross-fitted matched accuracy',table(header,macro_rows(main)),
        '## 학습 가중치·stopping criterion 민감도',table(header,macro_rows(['M1']+allp1)),
        '## Cell별 matched accuracy',
        '각 값은 accuracy (%) / 평균 출력 토큰입니다. 선택 fold에서의 SC16−1 pp 조건은 held-out fold의 정확도 보장이 아닙니다.',
        table(['모델','데이터','SC16 %']+[LABEL[m] for m in ['M1','asc']+CORE],point_rows(['M1','asc']+CORE,'cf_accuracy')),
        '## Dev에서 선택한 operating point',
        table(['모델','데이터','SC16 %']+[LABEL[m] for m in ['M1']+CORE],point_rows(['M1']+CORE,'accuracy')),
        '## 관측된 차이',*findings,
        '## 고정 예산 결과']
    for budget in CONFIG['selection']['budgets']:
        text += [f'### CF budget {budget}',table(header,macro_rows(main,f'cf_budget_{budget}'))]
    text += ['## 그림',
        'x축은 SC16 대비 출력 토큰 비율(log), y축은 test accuracy입니다. 회색 SC@k, ◆는 cross-fitted matched-accuracy point이며 색·축·frontier는 기존 plotting 함수를 사용합니다.',
        '![P1 Joint and Mixed VOI](figures/voi.png)',
        '![Prefix weighting](figures/weighting.png)',
        '![VOI and confidence criterion](figures/criterion.png)',
        '## 검증',
        f"고정 M1/SC의 재평가 최대 차이={max_baseline_error:.3g}. Native P1 posterior와 count lookup의 최대 차이={max_lookup:.3g}. 16개 응답 내 모든 유효 sufficient state의 coarse/fine log evidence 최대 차이={max_quad:.3g}. 파라미터·적분 해상도를 test 관측 전에 lock하고 source fit/bank/code/trace hash를 저장했습니다.",
        '새 구현은 반복 적분을 count sufficient-state lookup으로 바꿉니다. P1을 M1로 축소했을 때 기존 MC VOI와 일치하는지, native P1 posterior·미래 count 상태·label/suffix 격리·chunk 무관성·최소/최대 응답 수·strict VOI threshold·캐시 재사용을 검증했습니다.',
        '## 재현 커맨드',
        '```bash\npgm p1-stop trace --workers 4\npgm v2 method --plugin examples/p1_stopping.py\npgm v2 report --methods p1_independent_joint p1_linked_joint p1_independent_mixed p1_linked_mixed\npgm v2 plot --methods M1 asc p1_independent_joint p1_linked_joint p1_independent_mixed p1_linked_mixed --out results/p1_stopping/figures/voi\npgm p1-stop report\n```',
        '추가 그림도 같은 `pgm v2 plot`을 사용합니다. `weighting`의 methods: M1 p1_independent_mixed p1_linked_mixed p1_independent_mixed_v1 p1_linked_mixed_v1. `criterion`의 methods: M1 p1_independent_mixed p1_independent_mixed_confidence p1_linked_mixed p1_linked_mixed_confidence. `pgm p1-stop report`는 이 3개 그림을 PNG/SVG로 재생성합니다.',
        '## 範囲と限界',
        '이미 반복해서 탐색한 기존 bank에서의 비교입니다. 학습 목적·q 연결·prefix 가중치를 test에서 골라 단일 최적 모델로 보고하지 않았습니다. Mixed의 posterior 개선이 stopping 성능으로 이어지는지 별도 평가축에서 확인하며, 새 holdout 검증은 별도로 필요합니다.',
        '기존 exact-string 후보·xVerify 판정, 최대 16개 응답, 48 MC의 VOI 추정 오차, P1의 q/Normal 적분·bounded prior와 invalid 응답 생성 모델 부재가 남습니다. PGM 계산 시간·입력 토큰·특징 추출 비용은 출력 토큰 지표에 포함하지 않습니다.',
        '결과 파일: operating_points.csv(전체 선택 규칙), frontier.csv, macro_summary.csv, paired_vs_m1.csv, numerical_checks.csv, validation.json. 표준 method JSON은 results/v2/methods/p1_*/ 에 있습니다.']
    md='\n\n'.join(text)+'\n';(root/'README.md').write_text(md)
    from pgm_asc.v2.report import plot
    for name,methods in [
        ('voi',['M1','asc']+CORE),
        ('weighting',['M1','p1_independent_mixed','p1_linked_mixed','p1_independent_mixed_v1','p1_linked_mixed_v1']),
        ('criterion',['M1','p1_independent_mixed','p1_independent_mixed_confidence','p1_linked_mixed','p1_linked_mixed_confidence'])]:
        plot(methods,root/'figures'/name,title='P1 adaptive self-consistency: '+name)
    style='body{font-family:system-ui,sans-serif;max-width:1500px;margin:40px auto;padding:0 24px;line-height:1.65;color:#182331}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:8px;border-bottom:1px solid #dde2e8;text-align:left}th{background:#edf2f7}img{width:100%}pre{background:#f1f4f8;padding:15px;overflow:auto}'
    import html
    parts=md.split('```');rendered=[]
    for i,part in enumerate(parts):
        rendered.append('<pre><code>'+html.escape(part.removeprefix('bash\n').strip())+'</code></pre>' if i%2 else markdown_html(part))
    (root/'report.html').write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>P1 stopping</title><style>"+style+'</style><body>'+''.join(rendered)+'</body></html>')
    return str(root/'report.html')
