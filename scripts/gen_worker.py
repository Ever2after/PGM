"""Run pgm_asc.collection.study with an optional vLLM memory-fraction override.

pgm_asc/collection/study.py fixes gpu_memory_utilization=0.25 (the original study shared GPUs).
For Llama-3.1-8B on Omni-MATH that leaves a 63k-token KV cache for up to 64 x ~2.5k-token
sequences; the resulting preemption trips a vLLM 0.11 scheduler assertion on prompt logprobs.
Each worker here owns a whole GPU, so PGM_GPU_MEMORY_UTILIZATION raises the fraction. Sampling,
seeds and request identities are unchanged. Kept outside pgm_asc/ so lock source hashes stay valid.
"""
import os
import runpy
import sys

override = os.environ.get("PGM_GPU_MEMORY_UTILIZATION")
if override:
    import vllm

    class LLM(vllm.LLM):
        def __init__(self, *args, **kwargs):
            kwargs["gpu_memory_utilization"] = float(override)
            super().__init__(*args, **kwargs)

    vllm.LLM = LLM
sys.argv = ["pgm_asc.collection.study", *sys.argv[1:]]
runpy.run_module("pgm_asc.collection.study", run_name="__main__", alter_sys=True)
