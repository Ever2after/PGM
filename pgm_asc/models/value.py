"""Frozen continuation-payoff feature adaptation and Ridge prediction."""
from __future__ import annotations

import numpy as np

from pgm_asc.data.scaling import scaled


def value_x(observed,state_lp):
    return np.concatenate([observed['x'],np.exp(state_lp)],-1).astype(np.float64)


def ridge_predict(fitted,x):
    return fitted['model'].predict(scaled(x,fitted['scaler']))


def value_predictions(bundle,observed,state_lp):
    x=value_x(observed,state_lp);n=len(x)
    fixed=ridge_predict(bundle['fixed4'],x.reshape(-1,x.shape[-1])).reshape(n,8,2)
    fixed[:,:,1]=np.maximum(fixed[:,:,1]*1000,1)
    out={'fixed4':fixed}
    for lam,item in bundle['bellman'].items():
        q=np.zeros((n,8,2))
        for t,model in item['models'].items():
            p=ridge_predict(model,x[:,t]);q[:,t,0]=np.clip(p[:,0],0,1);q[:,t,1]=np.maximum(p[:,1]*1000,1)
        out['bellman_'+lam]=q
    return out
