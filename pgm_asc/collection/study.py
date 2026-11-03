"""TP1 replicas with cross-question batches; same reference sampling/parser."""
from __future__ import annotations
import argparse,copy,importlib.metadata,math,time
from pathlib import Path
import numpy as np
from filelock import FileLock
from pgm_asc.core.io import read_json,read_jsonl,write_json,utc_now,digest,sample_seed,log
from pgm_asc.core.integrity import verify_lock
from pgm_asc.collection.reasc_bank import render
from pgm_asc.policies.reasc_reference import confidence_groups,prediction


def extraction_prompt(question,response,dataset):
    instruction='Now, output only the numerical integer value of the answer. Final answer: ' if dataset=='gsm8k' else r'Now, output ONLY the answer in the form of $\boxed{{}}$. Therefore, the answer is'
    return [{'role':'user','content':question},{'role':'assistant','content':response},{'role':'user','content':instruction}]


def initialize(cfg):
    from vllm import LLM
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    path=snapshot_download(cfg['model_id'],revision=cfg['model_revision'],cache_dir=cfg['cache_dir'],local_files_only=True)
    maximum=64 if cfg['dataset']=='omnimath' else 128
    llm=LLM(model=path,tensor_parallel_size=1,dtype='bfloat16',max_model_len=8192,
            gpu_memory_utilization=.25,max_num_seqs=maximum,max_num_batched_tokens=8192,
            enforce_eager=False,enable_prefix_caching=True,seed=0,
            compilation_config={'level':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[x for x in [1,2,4,8,16,32,64,96,128] if x<=maximum]},
            **({'limit_mm_per_prompt':{'image':0}} if 'gemma-3' in cfg['model_id'] else {}))
    return llm,AutoTokenizer.from_pretrained(path)


def request_seed(qid,index,phase):
    return sample_seed(0,f'{qid}/{phase}',index)%(2**31-1)


def record(q,index,output,prompt,tok,seconds,identity,out):
    seq=output.outputs[0];ids=list(seq.token_ids)
    if len(ids)!=len(seq.logprobs):raise ValueError('Token/logprob length mismatch')
    kmax=max((len(d) for d in seq.logprobs),default=0)
    top_ids=np.full((len(ids),kmax),-1,np.int32);top_lp=np.full((len(ids),kmax),-np.inf,np.float32)
    scores=[];sampled=[]
    for i,(token,candidates) in enumerate(zip(ids,seq.logprobs)):
        if token not in candidates:raise ValueError('Missing actual sampled-token logprob')
        pairs=list(candidates.items());K=len(pairs)
        scores.append(-math.log(K)-sum(v.logprob for _,v in pairs)/K);sampled.append(float(candidates[token].logprob))
        top_ids[i,:K]=[k for k,_ in pairs];top_lp[i,:K]=[v.logprob for _,v in pairs]
    rid=f"{q['question_id']}__s{index:03d}";path=out/'top_logprobs'/f'{rid}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,token_ids=top_ids,logprobs=top_lp)
    c=confidence_groups(scores)
    return {'question_id':q['question_id'],'response_id':rid,'sample_index':index,'text':seq.text,
            'token_ids':ids,'prompt_token_ids':tok.encode(prompt,add_special_tokens=False),'native_sample_logprobs':sampled,
            'vllm_raw_logprobs':sampled,'self_certainty_per_token':scores,'lowest_confidence':c['lowest'],'bottom10_confidence':c['bottom10'],
            'generated_token_count':len(ids),'truncated':seq.finish_reason=='length','finish_reason':seq.finish_reason,
            'native_generation_seconds_share':seconds,'generation_identity':identity,'request_seed':request_seed(q['question_id'],index,'body')}


def collect(cfg,split,shard=0,shards=2,limit=None):
    root=Path(cfg['run_dir']);out=root/split;out.mkdir(parents=True,exist_ok=True)
    if split not in ['train','dev','calibration']:verify_lock(read_json(root/'selection_lock.json'))
    if not read_json(Path('artifacts/runs/reasc_pgm/reference_validation.json'))['passed']:raise ValueError('Reference parity required')
    qs=read_jsonl(Path(cfg['data_dir'])/f'{split}.jsonl')
    if limit is not None:qs=qs[:limit]
    qs=[q for i,q in enumerate(qs) if i%shards==shard]
    with FileLock(str(out/f'.generation_shard{shard}.lock'),timeout=0):
        pending=[q for q in qs if not (out/'questions'/f"{q['question_id']}.json").exists()]
        if not pending:return
        llm,tok=initialize(cfg);params=llm.get_default_sampling_params()
        for key in ['temperature','top_p','top_k','repetition_penalty']:
            if getattr(params,key)!=cfg[key]:raise ValueError(f'Reference sampling mismatch {key}')
        params.max_tokens=cfg['max_new_tokens'];params.logprobs=20;params.prompt_logprobs=0
        ep=llm.get_default_sampling_params();ep.max_tokens=128
        definition={'model_id':cfg['model_id'],'revision':cfg['model_revision'],'temperature':params.temperature,'top_p':params.top_p,'top_k':params.top_k,
                    'max_tokens':cfg['max_new_tokens'],'dataset':cfg['dataset'],'extraction_max_tokens':128,'logprobs':20,'vllm':importlib.metadata.version('vllm'),
                    'runtime':'TP1 replicas, decode graphs, maxseq64/batch4 Omni else maxseq128/batch8; memory.25',
                    'shards':shards,'request_seed':'hash(seed0,question_id,sample_index,body/inline/final) mod (2**31-1)',
                    'history_note':'Previously completed questions and calibration first responses are preserved with original identities; shared bank across all methods.'}
        identity=digest(definition);mp=out/'generation_manifests'/f'fast_shard{shard}.json'
        if mp.exists() and read_json(mp)['identity']!=identity:raise ValueError('Fast runtime changed')
        if not mp.exists():write_json(mp,definition|{'identity':identity,'shard':shard,'created_at':utc_now()})
        def generate(requests):
            if not requests:return []
            prompts=[render(tok,[{'role':'user','content':q['problem']}]) for q,i in requests]
            if any(len(tok.encode(p,add_special_tokens=False))+cfg['max_new_tokens']>8192 for p in prompts):raise ValueError('Context exceeded')
            sampling=[]
            for q,i in requests:
                p=copy.deepcopy(params);p.seed=request_seed(q['question_id'],i,'body');sampling.append(p)
            start=time.perf_counter();outputs=llm.generate(prompts,sampling,use_tqdm=False);elapsed=time.perf_counter()-start
            records=[record(q,i,o,p,tok,elapsed/len(requests),identity,out) for (q,i),o,p in zip(requests,outputs,prompts)]
            log('fast body batch',split=split,shard=shard,responses=len(records),seconds=elapsed,tokens=sum(r['generated_token_count'] for r in records))
            return records
        def extract(pairs,phase):
            if not pairs:return
            prompts=[render(tok,extraction_prompt(q['problem'],r['text'],cfg['dataset'])) for q,r in pairs]
            sampling=[]
            for q,r in pairs:
                p=copy.deepcopy(ep);p.seed=request_seed(q['question_id'],r['sample_index'],phase);sampling.append(p)
            start=time.perf_counter();outputs=llm.generate(prompts,sampling,use_tqdm=False);elapsed=time.perf_counter()-start
            for (q,r),p,o in zip(pairs,prompts,outputs):
                r['extraction_'+phase]={'text':o.outputs[0].text,'input_tokens':len(tok.encode(p,add_special_tokens=False)),
                    'output_tokens':len(o.outputs[0].token_ids),'seconds_share':elapsed/len(pairs),'request_seed':request_seed(q['question_id'],r['sample_index'],phase),'runtime_identity':identity}
        if split=='calibration':
            first=[q for q in qs if not (out/'first'/f"{q['question_id']}.json").exists()]
            for start in range(0,len(first),128):
                batch=first[start:start+128];records=generate([(q,0) for q in batch])
                for q,r in zip(batch,records):write_json(out/'first'/f"{q['question_id']}.json",r)
        for offset in range(0,len(pending),4 if cfg['dataset']=='omnimath' else 8):
            batch=pending[offset:offset+(4 if cfg['dataset']=='omnimath' else 8)];group={};requests=[];new=[]
            for q in batch:
                qid=q['question_id'];stage=out/'staged'/f'{qid}.json'
                if stage.exists():group[qid]=read_json(stage)['records'];continue
                group[qid]=[read_json(out/'first'/f'{qid}.json')] if split=='calibration' else []
                requests.extend((q,i) for i in range(1 if split=='calibration' else 0,16));new.append(q)
            for r in generate(requests):group[r['question_id']].append(r)
            pairs=[(q,r) for q in new for r in group[q['question_id']]]
            extract(pairs,'inline')
            for q in new:write_json(out/'staged'/f"{q['question_id']}.json",{'records':group[q['question_id']]})
            all_pairs=[(q,r) for q in batch for r in group[q['question_id']]]
            extract(all_pairs,'final')
            for q in batch:
                records=group[q['question_id']]
                if len(records)!=16 or [r['sample_index'] for r in records]!=list(range(16)):raise ValueError('Incomplete question/sample order')
                for r in records:
                    r['pred']=r['extraction_final']['text'].strip();r['prediction']=prediction(r['text'],r['pred'],cfg['dataset'])
                    chat=[{'role':'user','content':q['problem']},{'role':'assistant','content':r['text']}]
                    r['reference_chat_tokens']=len(tok.encode(tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=False),add_special_tokens=False))
                    write_json(out/'responses'/f"{r['response_id']}.json",r)
                write_json(out/'questions'/f"{q['question_id']}.json",{'question_id':q['question_id'],'responses':16,'generation_identity':identity,
                    'body_generation_identities':sorted({r['generation_identity'] for r in records}),'completed_at':utc_now()})
            done=sum((out/'questions'/f"{q['question_id']}.json").exists() for q in qs)
            write_json(out/f'generation_status_shard{shard}.json',{'state':'running','complete_questions':done,'total_questions':len(qs),'updated_at':utc_now()})
            log('fast question batch saved',split=split,shard=shard,complete_questions=done,total=len(qs))
        write_json(out/f'generation_status_shard{shard}.json',{'state':'complete','complete_questions':len(qs),'total_questions':len(qs),'updated_at':utc_now()})



def features(cfg,split,follow=False,limit=None):
    import torch
    from filelock import FileLock
    from transformers import AutoTokenizer,AutoModelForCausalLM,Gemma3ForConditionalGeneration
    from huggingface_hub import snapshot_download
    from pgm_asc.collection.teacher_forcing import score_batch,packed_batches
    root=Path(cfg['run_dir'])/split;root.mkdir(parents=True,exist_ok=True)
    if split not in ['train','dev','calibration']:verify_lock(read_json(Path(cfg['run_dir'])/'selection_lock.json'))
    qs=read_jsonl(Path(cfg['data_dir'])/f'{split}.jsonl')
    if limit is not None:qs=qs[:limit]
    names={f"{q['question_id']}__s{i:03d}" for q in qs for i in range(16)}
    torch.cuda.set_per_process_memory_fraction(.25)
    model=tok=None;completed=0;maxdiff=0.
    with FileLock(str(root/'.features.lock'),timeout=0):
        while True:
            records=[read_json(p) for p in sorted((root/'responses').glob('*.json')) if p.stem in names and not (root/'scored_responses'/p.name).exists()]
            if records and model is None:
                path=snapshot_download(cfg['model_id'],revision=cfg['model_revision'],cache_dir=cfg['cache_dir'],local_files_only=True)
                tok=AutoTokenizer.from_pretrained(path)
                cls=Gemma3ForConditionalGeneration if 'gemma-3' in cfg['model_id'] else AutoModelForCausalLM
                model=cls.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
                layers=getattr(model.config,'text_config',model.config).num_hidden_layers
            for batch in packed_batches(records,8,8192):
                torch.cuda.synchronize();start=time.perf_counter()
                output=score_batch(model,tok,batch,{'logit_chunk_size':128,'middle_layer':layers//2,'early_fraction':.8})
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                for r,vectors in output:
                    r['feature_seconds_share']=elapsed/len(batch);r['feature_input_tokens']=len(r['prompt_token_ids'])+len(r['token_ids'])
                    r['body_answer']=r['answer'];r['answer']=r['prediction']
                    maxdiff=max(maxdiff,r['raw_logprob_backend_mean_abs_difference'])
                    target=root/'embeddings'/f"{r['response_id']}.npz";target.parent.mkdir(exist_ok=True);np.savez_compressed(target,final_full=vectors['final_full'])
                    write_json(root/'scored_responses'/f"{r['response_id']}.json",r);completed+=1
            total=sum((root/'scored_responses'/f'{rid}.json').exists() for rid in names)
            write_json(root/'feature_status.json',{'state':'complete' if total==len(names) else 'waiting','responses':total,'total':len(names),'max_native_raw_lp_difference':maxdiff,'updated_at':utc_now()})
            if total==len(names) or not follow:return
            time.sleep(5)



def initialize_judge(cfg):
    from vllm import LLM
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    spec=cfg['judge'];path=snapshot_download(spec['model_id'],revision=spec['model_revision'],cache_dir=cfg['cache_dir'],local_files_only=True)
    llm=LLM(model=path,tensor_parallel_size=1,dtype='bfloat16',max_model_len=16384,gpu_memory_utilization=.25,
            max_num_seqs=64,max_num_batched_tokens=8192,enforce_eager=False,enable_prefix_caching=True,seed=0,
            compilation_config={'level':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4,8,16,32,64]})
    return llm,AutoTokenizer.from_pretrained(path)

from pgm_asc.collection.reasc_judge import literal,verdict
from pgm_asc.core.integrity import file_sha
from pgm_asc.policies.reasc_reference import exact_groups

def judge(cfg,split,follow=False,limit=None):
    from filelock import FileLock
    from vllm import SamplingParams
    root=Path(cfg['run_dir']);out=root/split
    if split not in ['train','dev','calibration']:verify_lock(read_json(root/'selection_lock.json'))
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
                llm,tok=initialize_judge(cfg)
                write_json(out/'judge_manifest.json',{'identity':identity,'spec':cfg['judge'],'template_sha256':file_sha(templatefile),'prompt_sha256':file_sha(promptfile),'preserve_reference_literal_backslash_n':True,'add_special_tokens':True,'decoding':'greedy','evaluation_only_cost':True,'created_at':utc_now()})
            for q in pending:
                qid=q['question_id'];records=[read_json(out/'responses'/f'{qid}__s{i:03d}.json') for i in range(16)]
                reps,ids=exact_groups([r['prediction'] for r in records])
                # One label per exact candidate plus full-response labels for the notebook metric.
                outputs=[r['text'] for r in records]+reps
                unique=list(dict.fromkeys(outputs));results={}
                prompts=[{'prompt_token_ids':tok.encode(template.format(query=prompt.format(question=q['problem'],output=s,answer=gold[qid])),add_special_tokens=True)} for s in unique]
                if any(len(p['prompt_token_ids'])+cfg['judge']['max_tokens']>16384 for p in prompts):raise ValueError('Judge context exceeded')
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
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','features','judge']);p.add_argument('--config',required=True);p.add_argument('--split',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2);p.add_argument('--limit',type=int);p.add_argument('--follow',action='store_true');a=p.parse_args();cfg=read_json(a.config)
    if a.command=='collect':collect(cfg,a.split,a.shard,a.shards,a.limit)
    elif a.command=='features':features(cfg,a.split,a.follow,a.limit)
    else:judge(cfg,a.split,a.follow,a.limit)
