"""Measure proposed TP1 graph/batching runtime on dev prompts only."""
import json,time
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from vllm import LLM


def main():
    cfg=json.loads(Path('configs/reasc_comparison.json').read_text())
    path=snapshot_download(cfg['model_id'],revision=cfg['model_revision'],cache_dir=cfg['cache_dir'],local_files_only=True)
    start=time.perf_counter()
    llm=LLM(model=path,tensor_parallel_size=1,dtype='bfloat16',max_model_len=8192,
            gpu_memory_utilization=.25,max_num_seqs=128,max_num_batched_tokens=8192,
            enforce_eager=False,enable_prefix_caching=True,seed=0,limit_mm_per_prompt={'image':0},
            compilation_config={'level':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4,8,16,32,64,96,128]})
    tok=AutoTokenizer.from_pretrained(path)
    qs=[json.loads(s) for s in (Path(cfg['data_dir'])/'dev.jsonl').read_text().splitlines()][:8]
    prompts=[tok.apply_chat_template([{'role':'user','content':q['problem']}],tokenize=False,add_generation_prompt=True) for q in qs]
    params=llm.get_default_sampling_params();params.max_tokens=1024;params.logprobs=20;params.prompt_logprobs=0
    report={'initialization_seconds':time.perf_counter()-start,'runtime':'TP1 graph FULL_DECODE_ONLY, level0, maxseq128, native logprobs20','cases':[]}
    for n in [16,128]:
        requests=[prompts[i//16] for i in range(n)]
        started=time.perf_counter();outputs=llm.generate(requests,params,use_tqdm=False);seconds=time.perf_counter()-started
        tokens=sum(len(o.outputs[0].token_ids) for o in outputs)
        result={'requests':n,'seconds':seconds,'output_tokens':tokens,'tokens_per_second':tokens/seconds,'seconds_per_response':seconds/n}
        report['cases'].append(result);print(json.dumps(result),flush=True)
    Path('artifacts/runs/reasc_pgm/runtime_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
