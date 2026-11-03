"""Fixed-horizon and Bellman payoff regression with outer-fold audits."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from pgm_asc.core.config import STATE as ROOT
from pgm_asc.core.integrity import load_tensor
from pgm_asc.core.io import log, write_json
from pgm_asc.data.scaling import scaled, scaler
from pgm_asc.data.state import costs, load, observed_features, selector_outputs, targets
from pgm_asc.models.state import StateModel, predict
from pgm_asc.models.value import ridge_predict, value_predictions, value_x
from pgm_asc.policies.stopping import policy_grid, stop_indices
from pgm_asc.training.recipes import name, recipes


def fit_ridge(x,y,alpha=10.):
    params=scaler(x)
    model=Ridge(alpha=alpha).fit(scaled(x,params),y)
    return {'scaler':params,'model':model}


def fit_bundle(observed,state_lp,correct,token_cost,index,lambdas):
    """Only index labels/observations fit the bundle; others are not consulted."""
    x=value_x(observed,state_lp)[index]
    reward=np.exp(state_lp[index,:,2])
    c=correct[index].astype(float);cost=token_cost[index]
    chosen=observed['chosen'][index]
    ts=np.arange(1,7);future=np.minimum(ts+4,7)
    y=np.stack([c[:,future]-c[:,ts],(cost[:,future]-cost[:,ts])/1000],-1)
    fixed4=fit_ridge(x[:,ts].reshape(-1,x.shape[-1]),y.reshape(-1,2))
    bellman={}
    for lam in lambdas:
        future_stop=np.full(len(index),7,dtype=int);models={}
        for t in range(6,0,-1):
            rr=np.arange(len(index))
            labels=np.stack([c[rr,future_stop],(cost[rr,future_stop]-cost[:,t])/1000],-1)
            fitted=fit_ridge(x[:,t],labels)
            prediction=ridge_predict(fitted,x[:,t])
            expected_reward=np.clip(prediction[:,0],0,1)
            expected_cost=np.maximum(prediction[:,1]*1000,1)
            continuing=(expected_reward-reward[:,t]>lam*expected_cost)|(chosen[:,t]<0)
            future_stop=np.where(continuing,future_stop,t)
            models[t]=fitted
        bellman[str(lam)]={'models':models,'lambda':lam}
    return {'fixed4':fixed4,'bellman':bellman,'train_indices':index.tolist(),
            'uses_realized_future_max':False,'reward_representation':'raw_current_state_R_probability'}


def fit_arm(recipe,observed,ycorrect,token_cost,config):
    directory=ROOT/'models'/name(recipe);out=directory/'policy_bundle.joblib'
    if out.exists():return
    with np.load(directory/'predictions.npz') as z:
        folds=z['folds'].copy();train_full=z['train_full'].copy();oof=z['train'].copy()
    bundles={};oof_values={}
    for f in range(5):
        saved=load_tensor(directory/f'fold{f}.pt')
        indices=np.flatnonzero(folds!=f);held=np.flatnonzero(folds==f)
        assert saved['train_indices']==indices.tolist() and not set(indices).intersection(held)
        model=StateModel(recipe['kind']);model.load_state_dict(saved['state_dict'])
        fold_lp=predict(model,observed,saved['scaler'])
        assert np.allclose(fold_lp[held],oof[held],atol=1e-6)
        bundles[f]=fit_bundle(observed,fold_lp,ycorrect,token_cost,indices,config['lambda_grid'])
        prediction=value_predictions(bundles[f],observed,fold_lp)
        for key,value in prediction.items():
            if key not in oof_values:oof_values[key]=np.empty_like(value)
            oof_values[key][held]=value[held]
    full=fit_bundle(observed,train_full,ycorrect,token_cost,np.arange(len(ycorrect)),config['lambda_grid'])
    joblib.dump({'full':full,'folds':bundles},out)
    np.savez_compressed(directory/'oof_policy_values.npz',**oof_values)
    # Entire outer-held policy decisions, labels used only below for audit metrics.
    policy_rows=[]
    for policy in policy_grid(config['lambda_grid'],[.5,.6,.7,.8,.85,.9,.925,.95,.975,.99,.995,None]):
        values=None
        if policy['family']=='fixed4':values=oof_values['fixed4']
        if policy['family']=='bellman':values=oof_values['bellman_'+str(policy['lambda'])]
        t,_=stop_indices(observed,oof,policy,values)
        qq=np.arange(len(t))
        policy_rows.append({'policy':policy,'OOF_accuracy':float(ycorrect[qq,t].mean()),
                            'OOF_tokens':float(token_cost[qq,t].mean()),'OOF_samples':float((t+1).mean())})
    write_json(directory/'oof_policy_audit.json',{'outer_folds':5,'entire_pipeline_question_overlap':0,
               'isolation_scope':'new_state_scaler_value_fitting_only; historical_frozen_v2_selection_used_v2_calibration',
               'state_fits_inside_outer_train':True,'scaler_and_value_fit_inside_outer_train':True,
               'in_sample_state_predictions_inside_value_training':True,
               'future_oracle_max_used':False,'policy_grid_not_selected_on_OOF':policy_rows})
    log('v5 honest outer-fold value policies complete',model=name(recipe))


def run(config):
    data,meta=load('train');outputs=selector_outputs(data,'train')
    for recipe in recipes():
        observed=observed_features(data,outputs[recipe['selector']]);_,correct=targets(data,observed)
        fit_arm(recipe,observed,correct,costs(meta),config)
