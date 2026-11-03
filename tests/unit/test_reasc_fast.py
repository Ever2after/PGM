"""Verify batching/sharding/resume without GPU or changing reference outputs."""
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from pgm_asc.core.io import write_json,write_jsonl,read_json
from pgm_asc.collection import reasc_fast as fast


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):return '|'.join(x['content'] for x in messages)
    def encode(self,text,**kwargs):return [1,2]


class Engine:
    def __init__(self):self.calls=[]
    def get_default_sampling_params(self):return NS(temperature=1.,top_p=.95,top_k=64,seed=None)
    def generate(self,prompts,params,**kwargs):
        self.calls.append((prompts,params))
        return [NS(outputs=[NS(text=r'\boxed{1}',token_ids=[7,8],logprobs=[{7:NS(logprob=-.2),8:NS(logprob=-1.2)}]*2,finish_reason='stop')]) for _ in prompts]


def config(tmp_path,split='dev',n=4):
    root=tmp_path/'run';data=tmp_path/'data'
    write_json(root/'reference_validation.json',{'passed':True})
    write_jsonl(data/f'{split}.jsonl',[{'question_id':f'q{i}','problem':f'problem{i}'} for i in range(n)])
    return {'run_dir':str(root),'data_dir':str(data),'model_id':'fake','model_revision':'fake',
            'temperature':1.,'top_p':.95,'top_k':64}


def test_replicas_disjoint_batched_and_resume(monkeypatch,tmp_path):
    cfg=config(tmp_path);e=Engine();monkeypatch.setattr(fast,'initialize',lambda cfg:(e,Tokenizer()))
    monkeypatch.setattr(fast.importlib.metadata,'version',lambda name:'fake-vllm')
    fast.collect(cfg,'dev',0,2)
    root=Path(cfg['run_dir'])/'dev'
    assert sorted(p.stem for p in (root/'questions').glob('*.json'))==['q0','q2']
    assert [len(c[0]) for c in e.calls]==[32,32,32]
    first=read_json(root/'responses/q0__s000.json');second=read_json(root/'responses/q0__s001.json')
    assert first['request_seed']!=second['request_seed']
    assert first['native_sample_logprobs']==[-.2,-1.2]
    with np.load(root/'top_logprobs/q0__s000.npz') as z:assert z['token_ids'].shape==(2,2)
    before=(root/'responses/q0__s000.json').read_bytes()
    fast.collect(cfg,'dev',1,2);calls=len(e.calls);fast.collect(cfg,'dev',0,2)
    assert len(e.calls)==calls and len(list((root/'questions').glob('*.json')))==4
    assert (root/'responses/q0__s000.json').read_bytes()==before


def test_calibration_preserves_existing_first_and_staged(monkeypatch,tmp_path):
    cfg=config(tmp_path,'calibration',2);root=Path(cfg['run_dir'])/'calibration'
    e=Engine();tok=Tokenizer();monkeypatch.setattr(fast,'initialize',lambda cfg:(e,tok))
    monkeypatch.setattr(fast.importlib.metadata,'version',lambda name:'fake-vllm')
    q={'question_id':'q0','problem':'problem0'}
    out=e.generate(['x'],[None])[0]
    root.mkdir();existing=fast.record(q,0,out,'x',tok,.1,'old-runtime',root)
    existing.pop('request_seed');write_json(root/'first/q0.json',existing)
    before=(root/'first/q0.json').read_bytes();e.calls=[]
    fast.collect(cfg,'calibration',0,1)
    assert [len(c[0]) for c in e.calls]==[1,30,32,32]
    assert (root/'first/q0.json').read_bytes()==before
    receipt=read_json(root/'questions/q0.json')
    assert 'old-runtime' in receipt['body_generation_identities']
    # A staged question can finish extraction without regenerating its bodies.
    (root/'questions/q0.json').unlink();e.calls=[]
    fast.collect(cfg,'calibration',0,1)
    assert [len(c[0]) for c in e.calls]==[16]
