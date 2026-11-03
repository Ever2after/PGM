"""Fetch pinned public assets for the supervised-structure follow-up study."""
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
from pgm_asc.core.io import read_json,write_json,utc_now,log


def prepare():
    pins=read_json('artifacts/runs/pgm_followup/source_pins.json')['sources'];cache='artifacts/hf_cache'
    gsm=snapshot_download('openai/gsm8k',repo_type='dataset',revision=pins['openai/gsm8k']['revision'],cache_dir=cache,
                          allow_patterns=['main/*.parquet','README.md'],token=False,max_workers=2)
    write_json('artifacts/runs/pgm_followup/gsm_asset.json',{'path':gsm,'revision':pins['openai/gsm8k']['revision']})
    def model(name):
        log('downloading pinned model',model=name)
        path=snapshot_download(name,revision=pins[name]['revision'],cache_dir=cache,token=False,max_workers=2,
                               allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','README.md','LICENSE'])
        log('pinned model available',model=name,path=path)
        return name,path
    with ThreadPoolExecutor(max_workers=2) as pool:paths=dict(pool.map(model,['Qwen/Qwen2.5-3B-Instruct','Qwen/Qwen2.5-7B-Instruct']))
    write_json('artifacts/runs/pgm_followup/assets_status.json',{'state':'complete','at':utc_now(),'models':paths,'gsm8k':gsm})


if __name__=='__main__':prepare()
