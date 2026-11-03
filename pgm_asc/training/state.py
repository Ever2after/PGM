"""Question-fold training of temporal state heads."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pgm_asc.core.config import STATE as ROOT
from pgm_asc.core.integrity import load_tensor
from pgm_asc.core.io import log, read_json, write_json
from pgm_asc.data.prefixes import training_weights
from pgm_asc.data.scaling import scaled, scaler
from pgm_asc.data.state import load, observed_features, selector_outputs, targets
from pgm_asc.evaluation.metrics import state_metrics
from pgm_asc.models.state import StateModel, predict
from pgm_asc.training.recipes import name, recipes


def fit(observed,y,train_indices,kind,seed,path,config):
    path=Path(path)
    if path.exists():
        saved=load_tensor(path)
        assert saved['train_indices']==train_indices.tolist()
        model=StateModel(kind);model.load_state_dict(saved['state_dict'])
        return model,saved['scaler']
    params=scaler(observed['x'][train_indices])
    x=torch.from_numpy(scaled(observed['x'][train_indices],params))
    inputs=[torch.from_numpy(observed[k][train_indices]) for k in ['new','changed','count']]
    label=torch.from_numpy(y[train_indices]);weights=training_weights()
    torch.manual_seed(seed)
    model=StateModel(kind)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['state_lr'],weight_decay=config['state_weight_decay'])
    rng=np.random.default_rng(seed)
    for epoch in range(config['state_epochs']):
        model.train();losses=[]
        for batch in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/config['state_batch_size']))):
            lp=model(x[batch],*[v[batch] for v in inputs])
            nll=F.nll_loss(lp.reshape(-1,3),label[batch].reshape(-1),reduction='none').reshape(-1,8)
            loss=(nll*weights).sum(1).mean()
            optimizer.zero_grad();loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite temporal state gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach()))
        if (epoch+1)%25==0:
            log('v5 state training',model=str(path),epoch=epoch+1,loss=float(np.mean(losses)))
            write_json(path.parent/'progress.json',{'model':str(path),'epoch':epoch+1})
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'kind':kind,'state_dict':model.state_dict(),'scaler':params,'seed':seed,
                'train_indices':train_indices.tolist(),'epochs':config['state_epochs']},path)
    return model,params


def train_arm(recipe,observed,y,dev,dev_y,config):
    out=ROOT/'models'/name(recipe);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return read_json(out/'result.json')
    n=len(y);seed=recipe['seed'];fold=np.empty(n,int)
    order=np.random.default_rng(seed).permutation(n);fold[order]=np.arange(n)%5
    oof=np.empty((n,8,3),float)
    for f in range(5):
        train=np.flatnonzero(fold!=f);held=np.flatnonzero(fold==f)
        assert not set(train).intersection(held)
        model,params=fit(observed,y,train,recipe['kind'],seed*100+f,out/f'fold{f}.pt',config)
        oof[held]=predict(model,observed,params,held)
    full,params=fit(observed,y,np.arange(n),recipe['kind'],seed*100+5,out/'full.pt',config)
    train_lp=predict(full,observed,params);dev_lp=predict(full,dev,params)
    np.savez_compressed(out/'predictions.npz',train=oof,train_full=train_lp,dev=dev_lp,folds=fold)
    result={'recipe':recipe,'name':name(recipe),'OOF_state':state_metrics(oof,y),'dev_state':state_metrics(dev_lp,dev_y),
            'parameters':sum(p.numel() for p in full.parameters()),'fits':6,
            'fold_question_overlap':0,'held_label_early_stopping':False}
    write_json(out/'result.json',result)
    return result


def run(config):
    torch.set_num_threads(2)
    train,tm=load('train');dev,dm=load('dev')
    traw=selector_outputs(train,'train');draw=selector_outputs(dev,'dev')
    inputs={}
    for s in ['M','S']:
        t=observed_features(train,traw[s]);d=observed_features(dev,draw[s])
        inputs[s]=(t,targets(train,t)[0],d,targets(dev,d)[0])
    result={}
    for recipe in recipes():
        write_json(ROOT/'training_status.json',{'state':'running','model':name(recipe),'completed_arms':len(result),'planned_arms':12})
        result[name(recipe)]=train_arm(recipe,*inputs[recipe['selector']],config)
    write_json(ROOT/'training_status.json',{'state':'complete','completed_arms':12,'neural_fits':72})
    write_json(ROOT/'training_results.json',result)
    return result
