"""Cached state-stage model/value predictions and frozen references."""
from __future__ import annotations

import joblib
import numpy as np

from pgm_asc.core.config import SELECTOR, STATE as ROOT
from pgm_asc.core.integrity import load_tensor
from pgm_asc.core.io import read_json
from pgm_asc.data.state import load, observed_features, selector_outputs
from pgm_asc.evaluation.replay import replay_policy
from pgm_asc.models.state import StateModel, predict
from pgm_asc.models.value import value_predictions
from pgm_asc.policies.baselines import read_split
from pgm_asc.policies.calibration import calibrate_state as calibrate
from pgm_asc.policies.continuation import continuation_predict
from pgm_asc.policies.stopping import distribution_history
from pgm_asc.training.recipes import name


class Context:
    def __init__(self,config,smoke=False):
        self.config=config;self.smoke=smoke
        self.out=ROOT/('smoke' if smoke else '.')
        self.data_cache={};self.raw_cache={};self.observed_cache={};self.state_cache={};self.value_cache={};self.bank_cache={}

    def data(self,split):
        if split not in self.data_cache:
            data,meta=load('train' if self.smoke else split)
            if self.smoke:
                # Tiny old-bank integration only. No fresh cohort or label access.
                data={k:v[:96] for k,v in data.items()};meta=meta[:96]
            self.data_cache[split]=(data,meta)
        return self.data_cache[split]

    def raw(self,split):
        if split not in self.raw_cache:
            data,_=self.data(split)
            all_outputs=selector_outputs(load('train')[0],'train') if self.smoke else selector_outputs(data,split)
            self.raw_cache[split]={k:v[:12] for k,v in all_outputs.items()} if self.smoke else all_outputs
        return self.raw_cache[split]

    def observed(self,selector,split):
        key=selector,split
        if key not in self.observed_cache:self.observed_cache[key]=observed_features(self.data(split)[0],self.raw(split)[selector])
        return self.observed_cache[key]

    def state(self,recipe,split):
        key=name(recipe),split
        if key not in self.state_cache:
            saved=load_tensor(ROOT/'models'/name(recipe)/'full.pt')
            model=StateModel(recipe['kind']);model.load_state_dict(saved['state_dict'])
            lp=predict(model,self.observed(recipe['selector'],split),saved['scaler'])
            self.state_cache[key]=lp
            out=self.out/'predictions'/name(recipe);out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/f'{split}.npz',logp=lp)
        return self.state_cache[key]

    def values(self,recipe,split):
        key=name(recipe),split
        if key not in self.value_cache:
            bundle=joblib.load(ROOT/'models'/name(recipe)/'policy_bundle.joblib')['full']
            self.value_cache[key]=value_predictions(bundle,self.observed(recipe['selector'],split),self.state(recipe,split))
        return self.value_cache[key]

    def banks(self,split):
        if split not in self.bank_cache:
            data,meta=self.data(split)
            if self.smoke:
                cfg=dict(self.config,run_dir=str(SELECTOR))
                self.bank_cache[split]=read_split(cfg,'calibration_a',meta)
            else:self.bank_cache[split]=read_split(self.config,split,meta)
        return self.bank_cache[split]

    def frozen(self,split):
        data,meta=self.data(split)
        old=read_json(SELECTOR/'selection_lock.json')['stages']['s6_continuation']['mixture']['selected']
        valid=data['candidate_mask'][:,:8].any(1).numpy().reshape(-1,8)
        history=distribution_history(calibrate(self.raw(split)['M'],old['calibrator']['T']),valid)
        continuation=continuation_predict(joblib.load(old['continuation_head']),history,data)
        return replay_policy(history,data,meta,old['policy'],continuation)
