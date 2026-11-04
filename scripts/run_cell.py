"""Run one follow-up cell end to end: fitting banks, fit/lock, evaluation banks, replay.

Transfer cells (config with `transfer_from`) skip fitting collection: their lock copies
the source cell's lock, and evaluation reuses whatever test bank is present (linked).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIT_SPLITS = ("calibration", "train", "dev")


def collect(config, cfg, split, gpus, limit=None):
    """Four workers on four GPUs: two generation shards, exact-token TF features, xVerify judge."""
    from pgm_asc.experiments.study_runner import bank_complete
    if limit is None and bank_complete(cfg, split):
        print(f"{split}: complete bank reused", flush=True)
        return
    logs = ROOT / cfg["run_dir"] / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tasks = (("generation_shard0", gpus[0], ["collect", "--shard", "0", "--shards", "2"]),
             ("generation_shard1", gpus[1], ["collect", "--shard", "1", "--shards", "2"]),
             ("features", gpus[2], ["features", "--follow"]),
             ("judge", gpus[3], ["judge", "--follow"]))
    workers = []
    try:
        for name, gpu, args in tasks:
            log = logs / f"{split}_{name}.log"
            offset = log.stat().st_size if log.exists() else 0
            handle = log.open("ab")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "4",
                   "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
            # Wrappers outside pgm_asc/: tf_worker.py supplies a missing pad token; gen_worker.py
            # applies an optional PGM_GPU_MEMORY_UTILIZATION override. The judge runs unwrapped.
            wrapper = {"features": "tf_worker.py", "judge": None}.get(name, "gen_worker.py")
            entry = [str(ROOT / "scripts" / wrapper)] if wrapper else ["-m", "pgm_asc.collection.study"]
            process = subprocess.Popen([sys.executable, "-u", *entry, *args,
                                        "--config", config, "--split", split,
                                        *(["--limit", str(limit)] if limit else [])],
                                       cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            workers.append({"name": name, "process": process, "handle": handle, "log": log, "offset": offset, "tail": b""})
        next_status = 0
        while True:
            for w in workers:
                # A CUDA engine can hang in teardown after an exception; inspect new log bytes.
                with w["log"].open("rb") as handle:
                    handle.seek(w["offset"])
                    data = w["tail"] + handle.read()
                    w["offset"] = handle.tell()
                w["tail"] = data[-64:]
                if b"Traceback (most recent call last)" in data:
                    raise RuntimeError(f"Worker traceback: {w['log']}")
                code = w["process"].poll()
                if code not in (None, 0):
                    raise RuntimeError(f"{w['name']} exited {code}: {w['log']}")
            if time.monotonic() >= next_status:
                root = ROOT / cfg["run_dir"] / split
                done = lambda d: len(list((root / d).glob("*.json"))) if (root / d).exists() else 0
                print(f"{split}: generated={done('questions')}, judged={done('judgments')}", flush=True)
                next_status = time.monotonic() + 60
            if all(w["process"].poll() == 0 for w in workers):
                break
            time.sleep(2)
        if limit is None and not bank_complete(cfg, split):
            raise RuntimeError(f"Workers exited with an incomplete {split} bank")
    finally:
        for w in workers:
            if w["process"].poll() is None:
                try:
                    os.killpg(w["process"].pid, signal.SIGTERM)
                    w["process"].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(w["process"].pid, signal.SIGKILL)
                    w["process"].wait()
                except ProcessLookupError:
                    pass
            w["handle"].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", default="0,1,2,3", help="two generation devices, TF device, judge device")
    parser.add_argument("--limit", type=int, help="smoke: collect only the first N questions of each fitting split, then stop")
    args = parser.parse_args()
    os.chdir(ROOT)
    from pgm_asc.core.io import read_json
    from pgm_asc.experiments.study_fit import ready, select
    from pgm_asc.evaluation.study import evaluate
    cfg = read_json(args.config)
    gpus = [int(v) for v in args.gpus.split(",")]
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.limit:
        collect(args.config, cfg, "calibration", gpus, limit=args.limit)
        print(f"{cfg['model_tag']}/{cfg['dataset']}: smoke collection complete; nothing fitted", flush=True)
        return
    if not cfg.get("transfer_from"):
        for split in FIT_SPLITS:
            collect(args.config, cfg, split, gpus)
        if not ready(cfg):
            raise RuntimeError("Fitting banks incomplete")
    started = time.perf_counter()
    select(cfg, args.config)
    print(f"{cfg['model_tag']}/{cfg['dataset']}: lock ready in {time.perf_counter() - started:.0f}s", flush=True)
    for split in cfg["evaluation_splits"]:
        if not cfg.get("transfer_from"):
            collect(args.config, cfg, split, gpus)
        result = evaluate(cfg, split)
        print(f"{cfg['model_tag']}/{cfg['dataset']}/{split}: evaluation complete, {result['questions']} questions", flush=True)


if __name__ == "__main__":
    main()
