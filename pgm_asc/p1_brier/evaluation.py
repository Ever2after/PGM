"""Repair/damage Brier and gain-error cross term, using identical forecasts."""
import numpy as np
from pgm_asc.diagnostics import metrics


def diagnostics(future,arrays,prefixes,horizons):
    hit=metrics.labels(arrays,metrics.reference(arrays))
    brier,cross,residual=[],[],[]
    for pi,t in enumerate(prefixes):
        for hi,h in enumerate(horizons):
            if t+h>16:continue
            yr=(~hit[:,t-1]&hit[:,t+h-1]).astype(float)
            yd=(hit[:,t-1]&~hit[:,t+h-1]).astype(float)
            er=future['repair'][:,pi,hi]-yr;ed=future['damage'][:,pi,hi]-yd
            gain_error=future['gain'][:,pi,hi]-(yr-yd)
            brier.append((er**2+ed**2)/2);cross.append(er*ed)
            residual.append(gain_error**2-(er**2+ed**2-2*er*ed))
    return dict(rd_mean_brier=np.mean(brier,axis=0),rd_error_cross=np.mean(cross,axis=0),
        gain_decomposition_residual=np.mean(residual,axis=0))
