"""Collect a full bank on one 4-GPU node: two generation shards, teacher forcing, and the judge run concurrently."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from pgm_asc.collect.workspace import Workspace
from pgm_asc.config import ARTIFACTS, ROOT
from pgm_asc.data.bank import load_cohort
from pgm_asc.utils import log


def collect_split(model, dataset, split, gpus=(0, 1, 2, 3), limit=None, memory=0.25, allow_test=False):
    """Run the four stage workers for one split and wait until every question is judged."""
    common = ["--model", model, "--dataset", dataset, "--split", split, "--memory", str(memory)]
    common += (["--limit", str(limit)] if limit else []) + (["--allow-test"] if allow_test else [])
    stages = [("generate_shard0", gpus[0], ["generate", "--shard", "0", "--shards", "2"]),
              ("generate_shard1", gpus[1], ["generate", "--shard", "1", "--shards", "2"]),
              ("features", gpus[2], ["features", "--follow"]),
              ("judge", gpus[3], ["judge", "--follow"])]
    logs = ARTIFACTS / "work" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    workers = []
    try:
        for name, gpu, args in stages:
            path = logs / f"{model}_{dataset}_{split}_{name}.log"
            handle = path.open("ab")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen([sys.executable, "-m", "pgm_asc", "collect", *args, *common], cwd=ROOT, env=env,
                                       stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            workers.append((name, process, handle, path))
        ws = Workspace(model, dataset, split)
        total = len(load_cohort(dataset, split)[:limit])
        next_status = 0.
        while True:
            for name, process, _, path in workers:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"{name} exited with {process.returncode}; see {path}")
            if all(process.poll() == 0 for _, process, _, _ in workers):
                break
            if time.monotonic() >= next_status:
                log("collecting", model=model, dataset=dataset, split=split, generated=len(ws.names("questions")),
                    judged=len(ws.names("judgments")), total=total)
                next_status = time.monotonic() + 60
            time.sleep(2)
    finally:
        for _, process, handle, _ in workers:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            handle.close()


def run_pipeline(model, dataset, gpus=(0, 1, 2, 3), memory=0.25, limit=None):
    """Fitting banks -> pack -> fit policy -> test bank -> pack -> evaluate (and MATH->Omni transfer when available)."""
    from pgm_asc.collect.pack import pack
    from pgm_asc.config import Cell, policy_path
    from pgm_asc.evaluate import evaluate_cell
    from pgm_asc.fit import fit_cell
    for split in ("calibration", "train", "dev"):
        collect_split(model, dataset, split, gpus, limit, memory)
    if limit:
        log("smoke collection complete; nothing packed or fitted", model=model, dataset=dataset)
        return
    pack(model, dataset, ("calibration", "train", "dev"))
    cell = Cell(model, dataset)
    if not policy_path(cell.policy_name).exists():
        fit_cell(cell)
    collect_split(model, dataset, "test", gpus, None, memory)
    pack(model, dataset)
    evaluate_cell(cell)
    if dataset == "omnimath" and policy_path(f"{model}_math").exists():
        evaluate_cell(Cell(model, "math_to_omnimath"))
