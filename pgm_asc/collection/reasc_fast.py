"""TP1 replicas with cross-question batches; same reference sampling/parser."""
from __future__ import annotations
import argparse,copy,importlib.metadata,math,time
from pathlib import Path
import numpy as np
from filelock import FileLock
from pgm_asc.core.io import read_json,read_jsonl,write_json,utc_now,digest,sample_seed,log
from pgm_asc.core.integrity import verify_lock
from pgm_asc.collection.reasc_bank import render,extraction_prompt
from pgm_asc.policies.reasc_reference import confidence_groups,prediction


def initialize(cfg):
    from vllm import LLM
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    path=snapshot_download(cfg['model_id'],revision=cfg['model_revision'],cache_dir=cfg['cache_dir'],local_files_only=True)
    llm=LLM(model=path,tensor_parallel_size=1,dtype='bfloat16',max_model_len=8192,
            gpu_memory_utilization=.25,max_num_seqs=128,max_num_batched_tokens=8192,
            enforce_eager=False,enable_prefix_caching=True,seed=0,limit_mm_per_prompt={'image':0},
            compilation_config={'level':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4,8,16,32,64,96,128]})
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
    if split in ['math500','fresh']:verify_lock(read_json(root/'selection_lock.json'))
    if not read_json(root/'reference_validation.json')['passed']:raise ValueError('Reference parity required')
    qs=read_jsonl(Path(cfg['data_dir'])/f'{split}.jsonl')
    if limit is not None:qs=qs[:limit]
    qs=[q for i,q in enumerate(qs) if i%shards==shard]
    with FileLock(str(out/f'.generation_shard{shard}.lock'),timeout=0):
        pending=[q for q in qs if not (out/'questions'/f"{q['question_id']}.json").exists()]
        if not pending:return
        llm,tok=initialize(cfg);params=llm.get_default_sampling_params()
        for key in ['temperature','top_p','top_k']:
            if getattr(params,key)!=cfg[key]:raise ValueError(f'Reference sampling mismatch {key}')
        params.max_tokens=1024;params.logprobs=20;params.prompt_logprobs=0
        ep=llm.get_default_sampling_params();ep.max_tokens=128
        definition={'model_id':cfg['model_id'],'revision':cfg['model_revision'],'temperature':params.temperature,'top_p':params.top_p,'top_k':params.top_k,
                    'max_tokens':1024,'extraction_max_tokens':128,'logprobs':20,'vllm':importlib.metadata.version('vllm'),
                    'runtime':'TP1 replicas; FULL_DECODE_ONLY graphs; compilation level0; maxseq128; question batch8; memory fraction.25',
                    'shards':shards,'request_seed':'hash(seed0,question_id,sample_index,body/inline/final) mod (2**31-1)',
                    'history_note':'Previously completed questions and calibration first responses are preserved with original identities; shared bank across all methods.'}
        identity=digest(definition);mp=out/'generation_manifests'/f'fast_shard{shard}.json'
        if mp.exists() and read_json(mp)['identity']!=identity:raise ValueError('Fast runtime changed')
        if not mp.exists():write_json(mp,definition|{'identity':identity,'shard':shard,'created_at':utc_now()})
        def generate(requests):
            if not requests:return []
            prompts=[render(tok,[{'role':'user','content':q['problem']}]) for q,i in requests]
            if any(len(tok.encode(p,add_special_tokens=False))+1024>8192 for p in prompts):raise ValueError('Context exceeded')
            sampling=[]
            for q,i in requests:
                p=copy.deepcopy(params);p.seed=request_seed(q['question_id'],i,'body');sampling.append(p)
            start=time.perf_counter();outputs=llm.generate(prompts,sampling,use_tqdm=False);elapsed=time.perf_counter()-start
            records=[record(q,i,o,p,tok,elapsed/len(requests),identity,out) for (q,i),o,p in zip(requests,outputs,prompts)]
            log('fast body batch',split=split,shard=shard,responses=len(records),seconds=elapsed,tokens=sum(r['generated_token_count'] for r in records))
            return records
        def extract(pairs,phase):
            if not pairs:return
            prompts=[render(tok,extraction_prompt(q['problem'],r['text'])) for q,r in pairs]
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
        for offset in range(0,len(pending),8):
            batch=pending[offset:offset+8];group={};requests=[];new=[]
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
                    r['pred']=r['extraction_final']['text'].strip();r['prediction']=prediction(r['text'],r['pred'])
                    chat=[{'role':'user','content':q['problem']},{'role':'assistant','content':r['text']}]
                    r['reference_chat_tokens']=len(tok.encode(tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=False),add_special_tokens=False))
                    write_json(out/'responses'/f"{r['response_id']}.json",r)
                write_json(out/'questions'/f"{q['question_id']}.json",{'question_id':q['question_id'],'responses':16,'generation_identity':identity,
                    'body_generation_identities':sorted({r['generation_identity'] for r in records}),'completed_at':utc_now()})
            done=sum((out/'questions'/f"{q['question_id']}.json").exists() for q in qs)
            write_json(out/f'generation_status_shard{shard}.json',{'state':'running','complete_questions':done,'total_questions':len(qs),'updated_at':utc_now()})
            log('fast question batch saved',split=split,shard=shard,complete_questions=done,total=len(qs))
        write_json(out/f'generation_status_shard{shard}.json',{'state':'complete','complete_questions':len(qs),'total_questions':len(qs),'updated_at':utc_now()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/reasc_comparison.json');p.add_argument('--split',required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=2);p.add_argument('--limit',type=int);a=p.parse_args()
    collect(read_json(a.config),a.split,a.shard,a.shards,a.limit)
