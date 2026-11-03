"""Pinned xVerify labels, isolated from policy inputs and test-time selection."""
from __future__ import annotations
import argparse, ast, time
from pathlib import Path
from pgm_asc.core.io import read_json,read_jsonl,write_json,utc_now,digest,log
from pgm_asc.core.integrity import file_sha,verify_lock
from pgm_asc.collection.reasc_bank import initialize
from pgm_asc.policies.reasc_reference import exact_groups


def literal(path,name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(name)


def verdict(text):
    value=text.strip().strip('[]').strip().lower()
    if value not in ['correct','incorrect']:
        raise ValueError('Invalid xVerify verdict: '+repr(text))
    return value=='correct'


def judge(cfg,split,follow=False,limit=None):
    from filelock import FileLock
    from vllm import SamplingParams
    root=Path(cfg['run_dir']);out=root/split
    if split in ['math500','fresh']:verify_lock(read_json(root/'selection_lock.json'))
    qs=read_jsonl(Path(cfg['data_dir'])/f'{split}.jsonl')
    if limit is not None:qs=qs[:limit]
    gold={q['question_id']:q['gold_answer'] for q in read_jsonl(Path(cfg['data_dir'])/f'{split}_gold.jsonl')}
    promptfile='third_party/xverify_official/src/xVerify/prompts/judge_prompt.py'
    templatefile='third_party/xverify_official/src/xVerify/prompts/base_template.py'
    prompt=literal(promptfile,'PROMPT');template=literal(templatefile,'BASE_TEMPLATE')['xVerify-3B-Ia']
    identity=digest({'judge':cfg['judge'],'prompt':file_sha(promptfile),'template':file_sha(templatefile),'add_special_tokens':True})
    llm=tok=None
    with FileLock(str(out/'.judge.lock'),timeout=0):
        while True:
            pending=[q for q in qs if (out/'questions'/f"{q['question_id']}.json").exists() and not (out/'judgments'/f"{q['question_id']}.json").exists()]
            if pending and llm is None:
                llm,tok=initialize(cfg,judge=True)
                write_json(out/'judge_manifest.json',{'identity':identity,'spec':cfg['judge'],'template_sha256':file_sha(templatefile),'prompt_sha256':file_sha(promptfile),'preserve_reference_literal_backslash_n':True,'add_special_tokens':True,'decoding':'greedy','evaluation_only_cost':True,'created_at':utc_now()})
            for q in pending:
                qid=q['question_id'];records=[read_json(out/'responses'/f'{qid}__s{i:03d}.json') for i in range(16)]
                reps,ids=exact_groups([r['prediction'] for r in records])
                # One label per exact candidate plus full-response labels for the notebook metric.
                outputs=[r['text'] for r in records]+reps
                unique=list(dict.fromkeys(outputs));results={}
                prompts=[{'prompt_token_ids':tok.encode(template.format(query=prompt.format(question=q['problem'],output=s,answer=gold[qid])),add_special_tokens=True)} for s in unique]
                if any(len(p['prompt_token_ids'])+cfg['judge']['max_tokens']>8192 for p in prompts):raise ValueError('Judge context exceeded')
                start=time.perf_counter()
                generations=llm.generate(prompts,SamplingParams(temperature=0.,max_tokens=cfg['judge']['max_tokens']),use_tqdm=False)
                for s,p,g in zip(unique,prompts,generations):
                    text=g.outputs[0].text
                    results[s]={'correct':verdict(text),'verdict':text,'input_tokens':len(p['prompt_token_ids']),'output_tokens':len(g.outputs[0].token_ids)}
                raw=[results[r['text']]['correct'] for r in records]
                notebook=[sum(raw[i] for i in range(16) if ids[i]==k)*2 >= int((ids==k).sum()) for k in range(len(reps))]
                write_json(out/'judgments'/f'{qid}.json',{'question_id':qid,'judge_identity':identity,'representatives':reps,'canonical_correct':[results[s]['correct'] for s in reps],
                    'raw_response_correct':raw,'reference_bank_majority_correct':notebook,'elapsed_seconds':time.perf_counter()-start,
                    'unique_judgments':len(unique),'input_tokens':sum(r['input_tokens'] for r in results.values()),'output_tokens':sum(r['output_tokens'] for r in results.values()),'details':[{'output':s,**r} for s,r in results.items()]})
                log('xVerify question saved',split=split,question_id=qid)
            completed=sum((out/'judgments'/f"{q['question_id']}.json").exists() for q in qs)
            write_json(out/'judge_status.json',{'state':'complete' if completed==len(qs) else 'waiting','questions':completed,'total':len(qs),'updated_at':utc_now()})
            if completed==len(qs) or not follow:return
            time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/reasc_comparison.json');p.add_argument('--split',required=True);p.add_argument('--follow',action='store_true');p.add_argument('--limit',type=int);a=p.parse_args()
    judge(read_json(a.config),a.split,a.follow,a.limit)
