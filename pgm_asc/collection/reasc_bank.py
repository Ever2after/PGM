"""Shared official-setting response/extraction bank; separate resumable stages."""
from __future__ import annotations
import argparse,os,time,math,importlib.metadata
from pathlib import Path
import numpy as np
from pgm_asc.core.io import read_json,read_jsonl,write_json,digest,utc_now,log
from pgm_asc.core.integrity import file_sha
from pgm_asc.policies.reasc_reference import confidence_groups,prediction


def extraction_prompt(question,response):
    return [{'role':'user','content':question},{'role':'assistant','content':response},
            {'role':'user','content':r'Now, output ONLY the answer in the form of $\boxed{{}}$. Therefore, the answer is'}]


def initialize(cfg,judge=False):
    import torch
    from vllm import LLM
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    spec=cfg['judge'] if judge else cfg
    path=snapshot_download(spec['model_id'],revision=spec['model_revision'],cache_dir=cfg['cache_dir'],local_files_only=True)
    llm=LLM(model=path,tokenizer=path,tensor_parallel_size=1 if judge else 2,
            dtype='bfloat16',max_model_len=8192,gpu_memory_utilization=.25,
            max_num_seqs=64 if judge else 32,max_num_batched_tokens=8192,
            enforce_eager=True,enable_prefix_caching=True,seed=0,
            **({} if judge else {'limit_mm_per_prompt':{'image':0}}))
    tokenizer=AutoTokenizer.from_pretrained(path)
    return llm,tokenizer


def render(tok,messages):
    return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)


def collect(cfg,split,limit=None):
    from filelock import FileLock
    root=Path(cfg['run_dir']);out=root/split;out.mkdir(parents=True,exist_ok=True)
    if split in ['math500','fresh'] and not (root/'selection_lock.json').exists():raise ValueError('Policy lock required before test generation')
    if not read_json(root/'reference_validation.json')['passed']:raise ValueError('Reference parity required')
    qs=read_jsonl(Path(cfg['data_dir'])/f'{split}.jsonl')
    if limit is not None:qs=qs[:limit]
    with FileLock(str(out/'.generation.lock'),timeout=0):
        if all((out/'questions'/f"{q['question_id']}.json").exists() for q in qs):return
        llm,tok=initialize(cfg)
        params=llm.get_default_sampling_params()
        defaults={k:getattr(params,k,None) for k in ['temperature','top_p','top_k','repetition_penalty','stop_token_ids','seed']}
        if any(defaults[k]!=cfg[k] for k in ['temperature','top_p','top_k']):raise ValueError('Resolved defaults differ: '+str(defaults))
        params.max_tokens=1024;params.logprobs=20;params.prompt_logprobs=0
        extract_params=llm.get_default_sampling_params();extract_params.max_tokens=128
        manifest={'created_at':utc_now(),'model_id':cfg['model_id'],'revision':cfg['model_revision'],'defaults':defaults,
                  'vllm':importlib.metadata.version('vllm'),'torch':importlib.metadata.version('torch'),
                  'transformers':importlib.metadata.version('transformers'),'max_new_tokens':1024,'num_samples':16,
                  'prompt_suffix':'','top_logprobs':20,'gpu_memory_utilization':.25,'tensor_parallel_size':2,'enforce_eager':True,
                  'operational_overrides':['max_model_len8192','max_num_seqs32','prefix_cache_enabled','eager_execution','shared_GPU_memory25%'],
                  'resume_rng_note':'Completed question files are reused; a resumed process restarts engine seed0; segment membership and start time are recorded.'}
        identity=digest({k:v for k,v in manifest.items() if k not in ['created_at','resume_rng_note']})
        old=out/'generation_manifest.json'
        if old.exists() and read_json(old)['identity']!=identity:raise ValueError('Generation definition changed')
        manifest['identity']=identity
        if not old.exists():write_json(old,manifest)
        def generate(q,count,start):
            prompt=render(tok,[{'role':'user','content':q['problem']}])
            prompt_ids=tok.encode(prompt,add_special_tokens=False)
            if len(prompt_ids)+1024>8192:raise ValueError('Prompt capacity exceeded')
            started=time.perf_counter();outputs=llm.generate([prompt]*count,params,use_tqdm=False);seconds=time.perf_counter()-started
            records=[]
            for i,output in enumerate(outputs):
                sample=output.outputs[0];ids=list(sample.token_ids);scores=[];lp=[]
                kmax=max((len(d) for d in sample.logprobs),default=0)
                top_ids=np.full((len(ids),kmax),-1,np.int32);top_lp=np.full((len(ids),kmax),-np.inf,np.float32)
                for t,(token,candidates) in enumerate(zip(ids,sample.logprobs)):
                    if not candidates or token not in candidates:raise ValueError('Missing sampled-token logprob')
                    values=list(candidates.items());K=len(values)
                    scores.append(-math.log(K)-sum(x.logprob for _,x in values)/K);lp.append(float(candidates[token].logprob))
                    top_ids[t,:K]=[a for a,_ in values];top_lp[t,:K]=[float(b.logprob) for _,b in values]
                index=start+i;rid=f"{q['question_id']}__s{index:03d}";path=out/'top_logprobs'/f'{rid}.npz';path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,token_ids=top_ids,logprobs=top_lp)
                g=confidence_groups(scores)
                records.append({'response_id':rid,'question_id':q['question_id'],'sample_index':index,'text':sample.text,
                                'token_ids':ids,'prompt_token_ids':prompt_ids,'native_sample_logprobs':lp,
                                'vllm_raw_logprobs':lp,'self_certainty_per_token':scores,'lowest_confidence':g['lowest'],'bottom10_confidence':g['bottom10'],
                                'generated_token_count':len(ids),'truncated':sample.finish_reason=='length','finish_reason':sample.finish_reason,
                                'native_generation_seconds_share':seconds/count,'generation_identity':identity})
            return records
        def extract(q,records):
            prompts=[render(tok,extraction_prompt(q['problem'],r['text'])) for r in records]
            if any(len(tok.encode(p,add_special_tokens=False))+128>8192 for p in prompts):raise ValueError('Extraction capacity exceeded')
            started=time.perf_counter();results=llm.generate(prompts,extract_params,use_tqdm=False);elapsed=time.perf_counter()-started
            return [{'text':r.outputs[0].text,'output_tokens':len(r.outputs[0].token_ids),
                     'input_tokens':len(tok.encode(p,add_special_tokens=False)),'seconds_share':elapsed/len(records)} for r,p in zip(results,prompts)]
        # The first calibration response is generated across different questions, as in the reference calibration runner.
        if split=='calibration':
            pending=[q for q in qs if not (out/'first'/f"{q['question_id']}.json").exists()]
            for offset in range(0,len(pending),16):
                chunk=pending[offset:offset+16]
                prompts=[render(tok,[{'role':'user','content':q['problem']}]) for q in chunk]
                started=time.perf_counter();outputs=llm.generate(prompts,params,use_tqdm=False);elapsed=time.perf_counter()-started
                for q,p,ro in zip(chunk,prompts,outputs):
                    seq=ro.outputs[0];ids=list(seq.token_ids);scores=[];samplelp=[];maxk=max(map(len,seq.logprobs),default=0)
                    ti=np.full((len(ids),maxk),-1,np.int32);tl=np.full((len(ids),maxk),-np.inf,np.float32)
                    for t,(token,d) in enumerate(zip(ids,seq.logprobs)):
                        pairs=list(d.items());K=len(pairs);scores.append(-math.log(K)-sum(v.logprob for _,v in pairs)/K);samplelp.append(float(d[token].logprob));ti[t,:K]=[k for k,_ in pairs];tl[t,:K]=[v.logprob for _,v in pairs]
                    g=confidence_groups(scores);rid=q['question_id']+'__s000';pp=out/'top_logprobs'/f'{rid}.npz';pp.parent.mkdir(exist_ok=True);np.savez_compressed(pp,token_ids=ti,logprobs=tl)
                    r={'response_id':rid,'question_id':q['question_id'],'sample_index':0,'text':seq.text,'token_ids':ids,
                       'prompt_token_ids':tok.encode(p,add_special_tokens=False),'native_sample_logprobs':samplelp,'vllm_raw_logprobs':samplelp,
                       'self_certainty_per_token':scores,'lowest_confidence':g['lowest'],'bottom10_confidence':g['bottom10'],
                       'generated_token_count':len(ids),'truncated':seq.finish_reason=='length','finish_reason':seq.finish_reason,
                       'native_generation_seconds_share':elapsed/len(chunk),'generation_identity':identity}
                    write_json(out/'first'/f"{q['question_id']}.json",r)
        for index,q in enumerate(qs):
            path=out/'questions'/f"{q['question_id']}.json"
            if path.exists():continue
            staging=out/'staged'/f"{q['question_id']}.json"
            if staging.exists():records=read_json(staging)['records']
            else:
                records=([read_json(out/'first'/f"{q['question_id']}.json")]+generate(q,15,1)) if split=='calibration' else generate(q,16,0)
                inline=extract(q,records)
                for r,e in zip(records,inline):r['extraction_inline']=e
                write_json(staging,{'records':records})
            # Final extraction is the overwrite pass performed by the reference shell.
            final=extract(q,records)
            for r,e in zip(records,final):
                r['extraction_final']=e;r['pred']=e['text'].strip();r['prediction']=prediction(r['text'],r['pred'])
                chat=[{'role':'user','content':q['problem']},{'role':'assistant','content':r['text']}]
                r['reference_chat_tokens']=len(tok.encode(tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=False),add_special_tokens=False))
                write_json(out/'responses'/f"{r['response_id']}.json",r)
            write_json(path,{'question_id':q['question_id'],'responses':16,'generation_identity':identity,'completed_at':utc_now()})
            write_json(out/'generation_status.json',{'state':'running','complete_questions':sum((out/'questions'/f"{x['question_id']}.json").exists() for x in qs),'total_questions':len(qs),'updated_at':utc_now()})
            log('reasc bank question saved',split=split,question=index+1,total=len(qs))
        write_json(out/'generation_status.json',{'state':'complete','complete_questions':len(qs),'responses':len(qs)*16,'updated_at':utc_now()})


def features(cfg,split,follow=False,limit=None):
    import torch
    from filelock import FileLock
    from transformers import AutoTokenizer,Gemma3ForConditionalGeneration
    from pgm_asc.collection.teacher_forcing import score_batch,packed_batches
    root=Path(cfg['run_dir'])/split;root.mkdir(parents=True,exist_ok=True)
    qs=read_jsonl(Path(cfg['data_dir'])/f'{split}.jsonl')
    if limit is not None:qs=qs[:limit]
    names={f"{q['question_id']}__s{i:03d}" for q in qs for i in range(16)}
    torch.cuda.set_per_process_memory_fraction(.25)
    model=tok=None;completed=0;maxdiff=0.
    with FileLock(str(root/'.features.lock'),timeout=0):
        while True:
            records=[read_json(p) for p in sorted((root/'responses').glob('*.json')) if p.stem in names and not (root/'scored_responses'/p.name).exists()]
            if records and model is None:
                path=Path(cfg['cache_dir'])/'models--google--gemma-3-4b-it/snapshots'/cfg['model_revision']
                tok=AutoTokenizer.from_pretrained(path);model=Gemma3ForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
            for batch in packed_batches(records,8,8192):
                torch.cuda.synchronize();start=time.perf_counter()
                output=score_batch(model,tok,batch,{'logit_chunk_size':128,'middle_layer':17,'early_fraction':.8})
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','features']);p.add_argument('--config',default='configs/reasc_comparison.json');p.add_argument('--split',required=True);p.add_argument('--limit',type=int);p.add_argument('--follow',action='store_true');a=p.parse_args();cfg=read_json(a.config)
    if a.command=='collect':collect(cfg,a.split,a.limit)
    else:features(cfg,a.split,a.follow,a.limit)
