"""Device environment for generation and feature-extraction subprocesses."""
from __future__ import annotations

import os
from pathlib import Path


def runtime_env(project, gpu, vllm=False):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false", "TMPDIR": str(project/"artifacts/tmp")})
    for key,name in [("VLLM_CACHE_ROOT","vllm_cache"),("TORCHINDUCTOR_CACHE_DIR","torchinductor_cache"),("TRITON_CACHE_DIR","triton_cache"),("CUDA_CACHE_PATH","cuda_cache")]:
        path=project/"artifacts"/name
        path.mkdir(parents=True,exist_ok=True)
        env[key]=str(path)
    if vllm:
        env["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
        env["VLLM_NO_USAGE_STATS"]="1"
        # Host Python/SQLite links Conda ICU, which requires this newer C++ ABI.
        cpp=Path("/opt/conda/lib/libstdc++.so.6")
        if cpp.exists():
            env["LD_PRELOAD"]=(str(cpp)+" "+env.get("LD_PRELOAD","")).strip()
    return env
