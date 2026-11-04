"""Resume frozen MATH-to-Omni transfer without editing any locked config."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODELS = ("qwen7b", "qwen3b", "gemma4b")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_bundle():
    manifest = read(ROOT / "omni_bundle_manifest.json")
    for name, info in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]:
            raise RuntimeError("Missing or changed bundle input: " + name)
    print(f"Verified {len(manifest['files'])} bundled files; Git snapshot {manifest['git_commit']}", flush=True)


def verify_locks(models):
    from pgm_asc.core.integrity import verify_lock
    for tag in models:
        for dataset in ("math", "omnimath"):
            path = ROOT / f"artifacts/runs/pgm_followup/{tag}_{dataset}/selection_lock.json"
            verify_lock(read(path))
        print(f"{tag}: original MATH and Omni transfer locks verified", flush=True)


def download(models):
    from huggingface_hub import snapshot_download
    assets = {}
    for tag in models:
        cfg = read(ROOT / f"configs/pgm_followup/{tag}_omnimath.json")
        for spec in (cfg, cfg["judge"]):
            assets[(spec["model_id"], spec["model_revision"])] = cfg["cache_dir"]
    for (model_id, revision), cache in assets.items():
        print(f"Downloading {model_id}@{revision}", flush=True)
        snapshot_download(model_id, revision=revision, cache_dir=cache,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
                          max_workers=4)


def collect(tag, gpus, limit=None):
    config = f"configs/pgm_followup/{tag}_omnimath.json"
    cfg = read(config)
    from pgm_asc.experiments.study_runner import bank_complete
    if limit is None and bank_complete(cfg, "test"):
        print(f"{tag}: existing complete Omni bank reused", flush=True)
        return
    out = ROOT / cfg["run_dir"] / "logs"
    out.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("generation_shard0", gpus[0], ["collect", "--shard", "0", "--shards", "2"]),
        ("generation_shard1", gpus[1], ["collect", "--shard", "1", "--shards", "2"]),
        ("features", gpus[2], ["features", "--follow"]),
        ("judge", gpus[3], ["judge", "--follow"]),
    ]
    workers = []
    try:
        for name, gpu, args in tasks:
            log = out / f"test_portable_{name}.log"
            offset = log.stat().st_size if log.exists() else 0
            handle = log.open("ab")
            command = [sys.executable, "-u", "-m", "pgm_asc.collection.study", *args,
                       "--config", config, "--split", "test"]
            if limit is not None:
                command += ["--limit", str(limit)]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
                   "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            workers.append({"name": name, "process": process, "handle": handle, "log": log,
                            "offset": offset, "tail": b""})
            print(f"{tag}/{name}: GPU {gpu}, PID {process.pid}, {log.relative_to(ROOT)}", flush=True)
        next_status = 0
        while True:
            for worker in workers:
                # A CUDA engine can hang during teardown after an exception.
                # Inspect only bytes appended by this invocation.
                with worker["log"].open("rb") as handle:
                    handle.seek(worker["offset"])
                    data = worker["tail"] + handle.read()
                    worker["offset"] = handle.tell()
                worker["tail"] = data[-64:]
                if b"Traceback (most recent call last)" in data:
                    raise RuntimeError(f"Worker traceback: {worker['log']}")
                code = worker["process"].poll()
                if code not in (None, 0):
                    raise RuntimeError(f"{worker['name']} exited {code}: {worker['log']}")
            if time.monotonic() >= next_status:
                root = ROOT / cfg["run_dir"] / "test"
                generated = len(list((root / "questions").glob("*.json")))
                judged = len(list((root / "judgments").glob("*.json")))
                print(f"{tag}: generated={generated}, judged={judged}, full cohort=999", flush=True)
                next_status = time.monotonic() + 30
            if all(worker["process"].poll() == 0 for worker in workers):
                break
            time.sleep(2)
        if limit is None and not bank_complete(cfg, "test"):
            raise RuntimeError("Workers exited with an incomplete Omni bank")
    finally:
        for worker in workers:
            process = worker["process"]
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
            worker["handle"].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "download", "run"))
    parser.add_argument("--model", choices=(*MODELS, "all"), default="qwen7b")
    parser.add_argument("--gpus", default="0,1,2,3", help="two generation devices, TF device, judge device")
    parser.add_argument("--offline", action="store_true", help="skip download; require pinned model cache")
    parser.add_argument("--limit", type=int, help="collection smoke only: first N questions, no evaluation")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    models = MODELS if args.model == "all" else (args.model,)
    verify_bundle()
    if args.command == "verify":
        # This path uses only the Python standard library; no GPU/dependencies.
        return
    verify_locks(models)
    if args.command == "download":
        download(models)
        return
    try:
        gpus = [int(value) for value in args.gpus.split(",")]
    except ValueError:
        parser.error("--gpus must contain four distinct nonnegative GPU indices")
    if len(gpus) != 4 or len(set(gpus)) != 4 or min(gpus) < 0:
        parser.error("--gpus must contain four distinct nonnegative GPU indices")
    if args.limit is not None and not 1 <= args.limit <= 999:
        parser.error("--limit must be between 1 and 999")
    if not args.offline:
        download(models)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    for tag in models:
        collect(tag, gpus, args.limit)
        if args.limit is None:
            from pgm_asc.evaluation.study import evaluate
            result = evaluate(read(f"configs/pgm_followup/{tag}_omnimath.json"), "test")
            print(f"{tag}: evaluation complete, {result['questions']} questions", flush=True)
        else:
            print(f"{tag}: collection smoke complete; no benchmark evaluation performed", flush=True)


if __name__ == "__main__":
    main()
