"""Baseline threshold families on the compact arrays (same semantics as ``pgm_asc.baselines``, the v1 replication).

Each family maps a threshold to one stopping index and chosen candidate per question:
SC@k (fixed k), ASC (Beta top-two confidence >= beta), ESC (first unanimous window of w), ReASC offline/online
(self-certainty-weighted ASC with the single-response gate). Defaults: SC16, ASC .95, ESC w=4, ReASC .95.
"""
from __future__ import annotations

import numpy as np

from pgm_asc.baselines import offline_calibrate, online_calibrate, trace
from pgm_asc.v2 import CONFIG

BOTTOM10, LOWEST = 0, 1
DEFAULTS = {"sc": 16, "asc": .95, "esc": 4, "reasc_offline": .95, "reasc_online": .95}


def majority(ids, end):
    return int(np.bincount(ids[:end]).argmax())


def sc_family(arrays):
    ids = arrays["ids"].astype(int)
    return {k: (np.full(len(ids), k - 1), np.array([majority(r, k) for r in ids])) for k in range(1, 17)}


def esc_family(arrays):
    ids = arrays["ids"].astype(int)
    out = {}
    for w in CONFIG["esc_windows"]:
        stop, chosen = [], []
        for r in ids:
            end = len(r)
            for start in range(0, len(r), w):
                if len(set(r[start:start + w])) == 1:
                    end = min(len(r), start + w)
                    break
            stop.append(end - 1)
            chosen.append(majority(r, end))
        out[w] = (np.array(stop), np.array(chosen))
    return out


def calibrations(data: dict, split: str) -> dict:
    cal = data["calibration"]
    offline = offline_calibrate(cal["x"][:, 0, LOWEST], cal["raw_correct"][:, 0])
    online = online_calibrate(data[split]["x"][:, 0, BOTTOM10])
    return {"reasc_offline": offline, "reasc_online": online}


def beta_family(arrays, method, calibration=None):
    ids = arrays["ids"].astype(int)
    scores = arrays["x"][..., BOTTOM10].astype(np.float64)
    kind = "asc" if method == "asc" else "reasc"
    traces = [trace(list(ids[i]), list(scores[i]), kind, calibration) for i in range(len(ids))]
    conf = np.stack([h["confidence"] for h in traces])
    chosen = np.stack([h["chosen"] for h in traces])
    out = {}
    for beta in CONFIG["gammas"]:
        hit = conf >= beta
        stop = np.where(hit.any(1), hit.argmax(1), ids.shape[1] - 1)
        out[beta] = (stop, chosen[np.arange(len(ids)), stop])
    return out


def families(data: dict, split: str) -> dict:
    """{method: {threshold: (stop index [Q], chosen slot [Q])}} for one split."""
    arrays = data[split]
    cal = calibrations(data, split)
    return {"sc": sc_family(arrays), "asc": beta_family(arrays, "asc"), "esc": esc_family(arrays),
            "reasc_offline": beta_family(arrays, "reasc", cal["reasc_offline"]),
            "reasc_online": beta_family(arrays, "reasc", cal["reasc_online"])}
