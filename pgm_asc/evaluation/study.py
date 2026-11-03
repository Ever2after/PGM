"""Locked, matched-supervision structure contrasts and cross-dataset transfer."""
from pathlib import Path
import time
import numpy as np
import torch
from pgm_asc.core.io import read_json,write_json,write_jsonl,utc_now
from pgm_asc.core.integrity import verify_lock
from pgm_asc.data.reasc_bank import load,build,traces
from pgm_asc.data.banks import scale
from pgm_asc.models.study import restore,predict
from pgm_asc.experiments.study_fit import attach,native_decision
from pgm_asc.policies.calibration import calibrate
from pgm_asc.policies.stopping import stopping_index,GAMMAS
from pgm_asc.policies.reasc_reference import decide,online_calibrate
from pgm_asc.evaluation.reasc_comparison import row as reference_row,metrics as reference_metrics,paired as reference_paired
from pgm_asc.evaluation.replay import replay_policy


def needs_tf(name):return not name.startswith('candidate_counts_')


def row(bank,index,chosen,reason,features,cfg):
    r=reference_row(bank,index,chosen,reason,features);n=cfg['rounded_parameters']
    r['output_tokens']=r['generated_tokens']+r['extraction_output_tokens']
    r['reference_proxy_T']=n*r['reference_chat_tokens']/1e12
    r['dense_2N_approx_T']=2*n*(r['reference_chat_tokens']+r['extraction_input_tokens']+r['extraction_output_tokens']+r['feature_input_tokens'])/1e12
    return r


def metrics(rows):
    return reference_metrics(rows)|{'mean_output_tokens':float(np.mean([r['output_tokens'] for r in rows])),
                                  'total_output_tokens':int(sum(r['output_tokens'] for r in rows)),
                                  'total_body_tokens':int(sum(r['generated_tokens'] for r in rows))}


def paired(a,b):
    x=reference_paired([r|{'reference_chat_tokens':r['output_tokens']} for r in a],[r|{'reference_chat_tokens':r['output_tokens']} for r in b])
    x['output_token_ratio']=x.pop('reference_cost_ratio');x['output_token_ratio_ci95']=x.pop('reference_cost_ratio_ci95')
    return x


def holm(pvalues):
    names=sorted(pvalues,key=pvalues.get);n=len(names);result={};previous=0.
    for rank,name in enumerate(names):
        value=min(1.,max(previous,pvalues[name]*(n-rank)));result[name]=value;previous=value
    return result


def evaluate(cfg,split):
    root=Path(cfg['run_dir']);lock=read_json(root/'selection_lock.json');verify_lock(lock);out=root/split
    if (out/'study_evaluation.json').exists():return read_json(out/'study_evaluation.json')
    torch.set_num_threads(4);started=time.perf_counter()
    banks=load(cfg,split);rep=read_json(lock['preprocessing']['representation']);scalers=read_json(lock['preprocessing']['scalers'])
    d,meta=build(banks,rep);d,meta=attach(scale(d,scalers),banks,meta,rep)
    offline=lock['reasc_offline'];online=online_calibrate([b['records'][0]['bottom10_confidence'] for b in banks]);rows={};summary={};curves={};prefix={}
    for method in ['pass1','sc16','asc','esc','reasc_offline','reasc_online','asc_dev_tuned','reasc_dev_tuned']:
        rr=[]
        for b in banks:
            if method.endswith('_dev_tuned'):
                kind=method.removesuffix('_dev_tuned');i,c,why=native_decision(b,kind,offline,lock['native_tuned'][kind]['selected']['gamma'])
            else:
                pred=[r['prediction'] for r in b['records']];scores=[r['bottom10_confidence'] for r in b['records']]
                i,c,why=decide(pred,scores,method,offline if method=='reasc_offline' else online if method=='reasc_online' else None)
                if method=='esc':why='esc_'+why
            rr.append(row(b,i,c,why,False,cfg))
        rows[method]=rr;summary[method]=metrics(rr)
    for name,spec in lock['methods'].items():
        model=restore(spec['checkpoint']);lp=calibrate(predict(model,d),d['t'],spec['calibrator']);history=traces(lp,d)
        prefix[name]={}
        for t in [2,4,8,16]:
            indices=np.arange(t-1,len(lp),16);chosen=lp[indices,:32].argmax(1);g=d['gold_groups'].numpy()
            prefix[name][str(t)]={'accuracy':float(g[indices,chosen].mean()),'nll_first_correct_or_OTHER':float(-lp[indices,d['target'].numpy()[indices]].mean())}
        for op in ['selected','fixed95']:
            policy=spec['selected']['policy'] if op=='selected' else spec['fixed95'];rr=[]
            for b,h in zip(banks,history):
                i,why=stopping_index(h,policy);rr.append(row(b,i,int(h['chosen'][i]),why,needs_tf(name),cfg))
            key=name+'__'+op;rows[key]=rr;summary[key]=metrics(rr)
        curves[name]=[]
        for gamma in GAMMAS:
            r=replay_policy(history,d,meta,{'minimum':2,'patience':1,'cap':16,'gamma':gamma})
            curves[name].append({'gamma':gamma,**{k:v for k,v in r.items() if k!='rows'}})
    contrasts={}
    mix=rows['mixture_s42__selected']
    for control in ['alpha1','random','candidate_confidence','deepsets_embedding']:
        contrasts[control]=paired(mix,rows[f'{control}_s42__selected'])
    adjusted=holm({k:v['mcnemar_exact_p'] for k,v in contrasts.items()})
    for k,v in contrasts.items():v['holm_p_four_primary_structure_contrasts']=adjusted[k]
    baseline={key:paired(mix,rows[key]) for key in ['sc16','reasc_offline','reasc_online','reasc_dev_tuned']}
    stability={}
    for arm in sorted({name.rsplit('_s',1)[0] for name in lock['methods']}):
        for op in ['selected','fixed95']:
            values=[summary[f'{arm}_s{s}__{op}'] for s in lock['train_seeds']]
            stability[f'{arm}__{op}']={k:{'mean':float(np.mean([v[k] for v in values])),'std_across_3_seeds':float(np.std([v[k] for v in values],ddof=1))} for k in ['correct','mean_output_tokens','samples']}
    result={'created_at':utc_now(),'model':cfg['model_id'],'dataset':cfg['dataset'],'split':split,'questions':len(banks),
            'evaluation_role':'new confirmatory MATH holdout' if split=='confirm' else 'previously observed exploratory benchmark' if cfg['model_tag']=='gemma4b' and split in ['math500','previous_fresh'] else 'zero-shot MATH-head transfer' if cfg.get('transfer_from') else 'new model/dataset evaluation under frozen protocol',
            'summary':summary,'primary_structure_contrasts_seed42':contrasts,'secondary_baseline_contrasts_seed42':baseline,'seed_stability':stability,
            'fixed_prefix_metrics':prefix,'predefined_threshold_curves':curves,'reference_online_calibration':online,'seconds':time.perf_counter()-started,
            'no_continuation':True,'primary_cost':'consumed output tokens including both extractions','supervision_budget':{'train':800,'dev':200,'calibration':128},
            'inference_cost_note':'Output-token savings exclude input/TF costs. Dense2N is a rounded-N approximation; bank replay is not causal serving latency. Count-only needs no TF at serving; experiment precomputes common feature banks.',
            'multiple_testing_note':'Holm correction covers four primary structure contrasts in this cell only; no global significance claim across cells or secondary operating points.'}
    for name,rr in rows.items():write_jsonl(out/'study_decisions'/f'{name}.jsonl',rr)
    write_json(out/'study_evaluation.json',result)
    lines=[f"# {cfg['model_tag']} / {cfg['dataset']} / {split}",'',result['evaluation_role'],'','Primary seed42. Every learned arm uses matched supervision, no continuation, min2/cap16.','',
           '| Method | Accuracy | Samples | Mean output tokens | Dense2N approx T |','|---|---:|---:|---:|---:|']
    for name,v in summary.items():
        if '_s' in name and '_s42__selected' not in name:continue
        lines.append(f"| {name} | {100*v['correct']:.2f}% | {v['samples']:.2f} | {v['mean_output_tokens']:.0f} | {v['dense_2N_approx_T']:.2f} |")
    lines+=['','## Fixed95, primary seed42','','| Arm | Accuracy | Samples | Mean output tokens |','|---|---:|---:|---:|']
    for name,v in summary.items():
        if '_s42__fixed95' in name:lines.append(f"| {name.removesuffix('_s42__fixed95')} | {100*v['correct']:.2f}% | {v['samples']:.2f} | {v['mean_output_tokens']:.0f} |")
    lines+=['','## Primary structure contrasts','','| Mixture minus control | Accuracy difference (pp) | Paired95% CI (pp) | Output ratio | Holm p |','|---|---:|---|---:|---:|']
    for name,v in contrasts.items():
        ci=v['accuracy_difference_ci95'];lines.append(f"| {name} | {100*v['accuracy_difference']:+.2f} | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | {v['output_token_ratio']:.3f} | {v['holm_p_four_primary_structure_contrasts']:.4f} |")
    lines+=['',result['inference_cost_note'],'',result['multiple_testing_note']]
    (out/'study_report.md').write_text('\n'.join(lines)+'\n')
    return result
