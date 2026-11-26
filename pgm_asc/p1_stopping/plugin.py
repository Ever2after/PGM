"""Expose locked P1 traces to the documented v2 method/score/report interface."""
from functools import lru_cache
import numpy as np

from pgm_asc.p1_stopping.run import definitions,out_dir,code_hashes
from pgm_asc.utils import file_sha256,read_json
from pgm_asc.v2 import CONFIG
from pgm_asc.v2.methods import register
from pgm_asc.v2.model import Spec
from pgm_asc.v2.run import arm_family


@lru_cache(maxsize=2)
def cached(model,dataset):
    root=out_dir();p=read_json(root/'protocol.json');c=read_json(root/'cells'/f'{model}_{dataset}.json')
    if p['v2']!=CONFIG or p['code_sha256']!=code_hashes():raise ValueError('Stopping evaluation configuration/code changed')
    if c['experiment_id']!=p['experiment_id']:raise ValueError('Stopping trace protocol mismatch')
    if file_sha256(c['source_fit_path'])!=c['source_fit_sha256']:raise ValueError('Frozen P1 fit changed')
    if file_sha256(c['bank_path'])!=c['bank_sha256']:raise ValueError('Frozen bank changed')
    for path,sha in c['baseline_files'].items():
        if file_sha256(path)!=sha:raise ValueError('Frozen v2 baseline changed')
    if file_sha256(c['trace_path'])!=c['trace_sha256']:raise ValueError('Stopping trace changed')
    with np.load(c['trace_path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}


def decision(arrays,threshold,context,definition):
    z=cached(context['model'],context['dataset']);s=context['split']
    for k in ('question_id','ids','valid'):
        if not np.array_equal(arrays[k],z[f'{s}.{k}']):raise ValueError('Stopping trace observation order differs')
    head=f"{definition['variant']}__{definition['training']}"
    tr={k:z[f'{s}.{head}.{k}'] for k in ('chosen','confidence','other','voi')}
    spec=Spec.of('M0' if definition['stop']=='confidence' else 'M1')
    return arm_family(tr,spec)[threshold]


def register_methods():
    for name,d in definitions().items():
        def fn(arrays,threshold,context,definition=d):return decision(arrays,threshold,context,definition)
        register(name,CONFIG['gammas'] if d['stop']=='confidence' else CONFIG['lambdas'],label=d['label'])(fn)
