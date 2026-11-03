"""Train-only feature normalization for state and value heads."""
from __future__ import annotations

import numpy as np


def scaler(x):
    v=np.asarray(x,dtype=float).reshape(-1,x.shape[-1]);std=v.std(0);std[std<1e-6]=1
    return {'mean':v.mean(0).tolist(),'std':std.tolist()}


def scaled(x,params):
    return ((x-np.asarray(params['mean']))/np.asarray(params['std'])).astype(np.float32)
