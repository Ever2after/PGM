"""Score the retained sklearn1.9.1 head in its original environment."""
from __future__ import annotations
import argparse
import joblib
import numpy as np
import sklearn


def main(head,inputs,outputs):
    if sklearn.__version__!='1.9.1':raise RuntimeError('Frozen head requires original sklearn1.9.1 environment')
    fitted=joblib.load(head)
    x=np.load(inputs,allow_pickle=False)
    values=fitted['model'].predict(x)*fitted['output_scale']
    values[:,1]=np.maximum(values[:,1],1.)
    np.save(outputs,values,allow_pickle=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('head');p.add_argument('inputs');p.add_argument('outputs');a=p.parse_args();main(a.head,a.inputs,a.outputs)
