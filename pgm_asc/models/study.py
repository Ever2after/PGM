"""Matched-budget structural controls, including causal embedding DeepSets."""
import copy,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pgm_asc.core.io import read_json,write_json,write_jsonl,log,utc_now
from pgm_asc.core.integrity import load_tensor
from pgm_asc.data.banks import INPUT_KEYS
from pgm_asc.models.selector import PGM

ARMS={
    'mixture':{'mode':'mixture'},'uniform':{'mode':'uniform'},'random':{'mode':'random'},
    'alpha1':{'mode':'mixture','alpha':1.},'single_partition':{'mode':'single','partition':1},
    'all_one':{'mode':'single','partition':0},'singletons':{'mode':'single','partition':5},
    'candidate_counts':{'mode':'candidate_mlp','count_only':True},
    'candidate_confidence':{'mode':'candidate_mlp'},'deepsets_confidence':{'mode':'deepsets'},
    'deepsets_embedding':{'mode':'deepsets','use_embeddings':True}}


def make(spec,embedding_dim):
    p={k:v for k,v in spec.items() if k not in ['count_only','use_embeddings']}
    if spec.get('use_embeddings'):p['embedding_dim']=embedding_dim
    model=PGM(32,**p);model.count_only=spec.get('count_only',False);return model


def batch(data,idx,model):
    b={k:data[k][idx] for k in INPUT_KEYS}
    if model.count_only:
        b['candidate']=b['candidate'].clone();b['candidate'][:,:,2:]=0
        b['w']=b['w'].clone();b['w'][:,8:]=0
    if model.embedding_dim:
        q,inverse=torch.unique(data['question_index'][idx],return_inverse=True)
        b.update(unique_h=data['embeddings'][q],unique_x=data['x'][q*16+15],response_owner=inverse)
    return b


def predict(model,data):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(data['t']),512):out.append(model(batch(data,slice(start,start+512),model)).numpy())
    return np.concatenate(out)


def restore(path):
    s=load_tensor(path);m=make(s['spec'],s['embedding_dim']);m.load_state_dict(s['state_dict']);m.eval();return m


def fit(train,dev,spec,cfg,root,seed):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'fit.json').exists():return restore(root/'checkpoint.pt')
    torch.set_num_threads(4);torch.manual_seed(seed)
    dim=train['embeddings'].shape[-1];model=make(spec,dim);p=cfg['train']
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    weights=torch.tensor([.2,.2]+[.1]*2+[.05]*4+[.025]*8)
    rng=np.random.default_rng(seed);nq=len(train['t'])//16;best=float('inf');bad=0;history=[];started=time.perf_counter()
    active=None
    for epoch in range(p['epochs']):
        model.train();order=rng.permutation(nq);losses=[]
        for start in range(0,nq,p['question_batch']):
            idx=torch.from_numpy((order[start:start+p['question_batch'],None]*16+np.arange(16)).reshape(-1))
            lp=model(batch(train,idx,model));loss=(F.nll_loss(lp,train['target'][idx],reduction='none').reshape(-1,16)*weights).sum(1).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad();loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError('Nonfinite gradient')
            if active is None:active=sum(int((v.grad!=0).sum()) for v in model.parameters() if v.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        lp=predict(model,dev);target=dev['target'].numpy()
        score=float((-lp[np.arange(len(lp)),target].reshape(-1,16)*weights.numpy()).sum(1).mean())
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'dev_five_bucket_nll':score})
        if score<best-1e-7:best=score;bad=0;best_epoch=epoch+1;state=copy.deepcopy(model.state_dict())
        else:bad+=1
        write_json(root/'progress.json',history[-1]|{'best_epoch':best_epoch,'updated_at':utc_now()})
        if epoch==0 or (epoch+1)%20==0:log('study fit',arm=root.parent.name,seed=seed,epoch=epoch+1,dev_nll=score)
        if bad>=p['patience']:break
    model.load_state_dict(state)
    torch.save({'state_dict':state,'spec':spec,'embedding_dim':dim,'seed':seed},root/'checkpoint.pt')
    write_jsonl(root/'history.jsonl',history)
    write_json(root/'fit.json',{'seed':seed,'spec':spec,'epochs':len(history),'best_epoch':best_epoch,'dev_nll':best,
               'total_parameters_including_dormant_heads':sum(v.numel() for v in model.parameters()),
               'nonzero_gradient_parameter_entries_first_step':active,'seconds':time.perf_counter()-started,'alpha':float(model.alpha.detach())})
    return model
