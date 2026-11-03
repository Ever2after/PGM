"""Locked-policy replay, two label profiles, costs and paired question CIs."""
from __future__ import annotations
import argparse,time,subprocess
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import binomtest
from pgm_asc.core.io import read_json,write_json,write_jsonl,utc_now
from pgm_asc.core.integrity import verify_lock
from pgm_asc.data.reasc_bank import load,build,traces
from pgm_asc.data.banks import scale
from pgm_asc.models.selector import predict
from pgm_asc.experiments.reasc_comparison import restore
from pgm_asc.policies.calibration import calibrate
from pgm_asc.policies.continuation import continuation_predict,continuation_features
from pgm_asc.policies.stopping import stopping_index
from pgm_asc.policies.reasc_reference import decide,online_calibrate,exact_groups


def metrics(rows):
    keys=['correct','reference_correct','samples','reference_chat_tokens','generated_tokens','extraction_input_tokens','extraction_output_tokens',
          'feature_input_tokens','feature_seconds','native_generation_seconds','extraction_seconds','reference_proxy_T','dense_2N_approx_T']
    ci=binomtest(sum(r['correct'] for r in rows),len(rows)).proportion_ci(confidence_level=.95,method='exact')
    return {'questions':len(rows),'accuracy_ci95_exact':[float(ci.low),float(ci.high)],**{k:float(np.mean([r[k] for r in rows])) for k in keys}}


def row(bank,index,chosen,reason,is_pgm):
    reps,ids=exact_groups([r['prediction'] for r in bank['records']]);t=index+1;consumed=bank['records'][:t]
    if chosen<0 or chosen>=len(reps):raise ValueError('Selected unseen answer')
    labels=bank['labels'];candidate=reps[chosen];correct=labels['canonical_correct'][chosen]
    # ESC alone uses the full bank in the released notebook's metric aggregation.
    verdict_end=16 if reason.startswith('esc_') else t
    raw=[labels['raw_response_correct'][i] for i in range(verdict_end) if ids[i]==chosen]
    reference_correct=sum(raw)*2>=len(raw)
    chat=sum(r['reference_chat_tokens'] for r in consumed);gen=sum(r['generated_token_count'] for r in consumed)
    extraction_in=sum(r[k]['input_tokens'] for r in consumed for k in ['extraction_inline','extraction_final'])
    extraction_out=sum(r[k]['output_tokens'] for r in consumed for k in ['extraction_inline','extraction_final'])
    tf=sum(r['feature_input_tokens'] for r in consumed) if is_pgm else 0
    # Notebook uses rounded4B for Gemma3-4B; 2N approximation is explicitly separate.
    return {'question_id':bank['question']['question_id'],'answer':candidate,'chosen':chosen,'samples':t,'reason':reason,'correct':bool(correct),'reference_correct':bool(reference_correct),
            'reference_chat_tokens':chat,'generated_tokens':gen,'extraction_input_tokens':extraction_in,'extraction_output_tokens':extraction_out,
            'feature_input_tokens':tf,'feature_seconds':sum(r['feature_seconds_share'] for r in consumed) if is_pgm else 0.,
            'native_generation_seconds':sum(r['native_generation_seconds_share'] for r in consumed),
            'extraction_seconds':sum(r[k]['seconds_share'] for r in consumed for k in ['extraction_inline','extraction_final']),
            'reference_proxy_T':4e9*chat/1e12,'dense_2N_approx_T':2*4e9*(chat+extraction_in+extraction_out+tf)/1e12}


def paired(a,b,seed=20261102):
    if [r['question_id'] for r in a]!=[r['question_id'] for r in b]:raise ValueError('Unpaired cohorts')
    aa=np.array([r['correct'] for r in a],int);bb=np.array([r['correct'] for r in b],int)
    ac=np.array([r['reference_chat_tokens'] for r in a]);bc=np.array([r['reference_chat_tokens'] for r in b])
    ad=np.array([r['dense_2N_approx_T'] for r in a]);bd=np.array([r['dense_2N_approx_T'] for r in b])
    rng=np.random.default_rng(seed);delta=[];ratio=[];dense=[]
    for _ in range(10):
        idx=rng.integers(0,len(a),(1000,len(a)));delta.extend((aa[idx]-bb[idx]).mean(1));ratio.extend(ac[idx].mean(1)/bc[idx].mean(1));dense.extend(ad[idx].mean(1)/bd[idx].mean(1))
    win=int(((aa==1)&(bb==0)).sum());lose=int(((aa==0)&(bb==1)).sum());ci=np.quantile(delta,[.025,.975]).tolist()
    return {'accuracy_difference':float((aa-bb).mean()),'accuracy_difference_ci95':ci,'reference_cost_ratio':float(ac.mean()/bc.mean()),'reference_cost_ratio_ci95':np.quantile(ratio,[.025,.975]).tolist(),
            'dense_2N_ratio':float(ad.mean()/bd.mean()),'dense_2N_ratio_ci95':np.quantile(dense,[.025,.975]).tolist(),'pgm_only_correct':win,'baseline_only_correct':lose,
            'mcnemar_exact_p':float(binomtest(win,win+lose,.5).pvalue) if win+lose else 1.,'bootstrap_replicates':10000,'unit':'question',
            'practical_target_pass':bool((aa-bb).mean()>=-.01-1e-12 and ac.mean()/bc.mean()<=.9),'statistical_noninferiority_1pp':bool(ci[0]>=-.01-1e-12)}


def evaluate(cfg,split):
    root=Path(cfg['run_dir']);lock=read_json(root/'selection_lock.json');verify_lock(lock)
    banks=load(cfg,split);representation=read_json(root/'representation.json');scalers=read_json(root/'scalers.json')
    raw,meta=build(banks,representation);data=scale(raw,scalers)
    offline=read_json(root/'reasc_offline_calibration.json');online=online_calibrate([b['records'][0]['bottom10_confidence'] for b in banks])
    write_json(root/split/'reasc_online_calibration.json',online);rows={}
    start=time.perf_counter()
    for method in ['pass1','sc16','asc','esc','reasc_offline','reasc_online']:
        rows[method]=[]
        for b in banks:
            pred=[r['prediction'] for r in b['records']];scores=[r['bottom10_confidence'] for r in b['records']]
            idx,chosen,reason=decide(pred,scores,method,offline if method=='reasc_offline' else online if method=='reasc_online' else None)
            rows[method].append(row(b,idx,chosen,'esc_'+reason if method=='esc' else reason,False))
    baseline_head_seconds=time.perf_counter()-start;head_times={}
    for name,spec in lock['methods'].items():
        if name=='frozen_v2_transfer':
            d,_=build(banks,read_json(root/'frozen_representation.json'));d=scale(d,read_json(root/'frozen_scalers.json'))
        else:d=data
        model=restore(spec['checkpoint']);started=time.perf_counter();lp,_=predict(model,d)
        history=traces(calibrate(lp,d['t'],spec['calibrator']),d)
        if name=='frozen_v2_transfer':
            cache=root/split/'transfer_cache';cache.mkdir(parents=True,exist_ok=True)
            inputs=cache/'features.npy';outputs=cache/'values.npy'
            np.save(inputs,continuation_features(history,d),allow_pickle=False)
            subprocess.run([str(Path('.venv/bin/python').absolute()),'-m','pgm_asc.experiments.reasc_transfer',spec['continuation_head'],str(inputs),str(outputs)],check=True)
            values=np.load(outputs,allow_pickle=False).reshape(len(history),16,2)
        else:
            values=continuation_predict(joblib.load(spec['continuation_head']),history,d) if spec.get('continuation_head') else None
        rows[name]=[]
        for qi,(b,h) in enumerate(zip(banks,history)):
            idx,reason=stopping_index(h,spec['policy'],None if values is None else values[qi]);rows[name].append(row(b,idx,int(h['chosen'][idx]),reason,True))
        head_times[name]=time.perf_counter()-started
    summary={m:metrics(r) for m,r in rows.items()}
    comparisons={b:paired(rows['pgm_mixture'],rows[b]) for b in ['reasc_offline','reasc_online','sc16']}
    result={'created_at':utc_now(),'split':split,'protocol':cfg['label'],'primary_metric':'canonical candidate xVerify','summary':summary,'pgm_mixture_vs':comparisons,
            'head_replay_seconds':head_times,'all_baseline_replay_and_GMM_seconds':baseline_head_seconds,
            'cost_note':'Bank replay; measured shares are offline collection costs, not causal serving latency. Dense2N is a rounded4B approximation; no end-to-end FLOPs profiler claim.',
            'setup_costs':{'selection_seconds':lock['elapsed_seconds'],'fitting_questions':lock['supervision_questions'],'judge_seconds':sum(b['labels']['elapsed_seconds'] for b in banks),
                           'judge_input_tokens':sum(b['labels']['input_tokens'] for b in banks),'judge_output_tokens':sum(b['labels']['output_tokens'] for b in banks),
                           'training_calibration_collection':lock['setup_costs']},
            'online_first_pass':{'questions':len(banks),'chat_tokens':sum(b['records'][0]['reference_chat_tokens'] for b in banks),'transductive_barrier':True},
            'ambiguous_multi_correct_questions':sum(sum(b['labels']['canonical_correct'])>1 for b in banks)}
    for method,r in rows.items():write_jsonl(root/split/'decisions'/f'{method}.jsonl',r)
    write_json(root/split/'evaluation.json',result)
    lines=[f'# ReASC vs PGM: {split}', '',cfg['label'],'','Primary metric: selected candidate judged by pinned xVerify. Reference metric uses raw-response verdict votes.','',
           '| Method | Accuracy | Reference accuracy | Samples | Chat tokens | Reference proxy T | Dense 2N approx T |','|---|---:|---:|---:|---:|---:|---:|']
    for m,r in summary.items():lines.append(f"| {m} | {100*r['correct']:.2f}% | {100*r['reference_correct']:.2f}% | {r['samples']:.2f} | {r['reference_chat_tokens']:.0f} | {r['reference_proxy_T']:.2f} | {r['dense_2N_approx_T']:.2f} |")
    lines+=['','PGM mixture vs ReASC offline:','', '```json',__import__('json').dumps(comparisons['reasc_offline'],indent=2),'```','',result['cost_note']]
    (root/split/'report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/reasc_comparison.json');p.add_argument('--split',required=True);a=p.parse_args();evaluate(read_json(a.config),a.split)
