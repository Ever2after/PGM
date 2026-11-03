"""vLLM-only response generation in its isolated environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import time
from pathlib import Path

from pgm_asc.core.answers import boxed_span
from pgm_asc.core.config import runtime_config
from pgm_asc.core.io import digest, log, read_json, read_jsonl, sample_seed, utc_now, write_json


def split_root(config, split):
    return Path(config["run_dir"]) / split


def generate(config, split):
    from filelock import FileLock
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from huggingface_hub import snapshot_download

    root = split_root(config, split)
    output_dir = root / "responses"
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(Path(config["data_dir"]) / f"{split}.jsonl")
    samples = config["samples_per_question"]
    fields = ["model_id", "model_revision", "seed", "max_new_tokens", "temperature", "top_p", "top_k", "dtype", "max_model_len", "gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens", "request_chunk_size", "prompt_suffix"]
    manifest = {"backend": "vllm", "backend_version": importlib.metadata.version("vllm"), "generation_config": {k: config[k] for k in fields}, "split": split, "question_hash": digest(questions), "samples_per_question": samples, "confidence_mode": "raw_logprobs", "sampling_seed_scheme": "sha256(base_seed,question_id,sample_index)", "implementation": "vllm_token_ids_v1"}
    fingerprint = digest(manifest)
    manifest_path = root / "generation_manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["fingerprint"] != fingerprint:
        raise ValueError("Existing vLLM run has different configuration or cohort")
    if not manifest_path.exists():
        write_json(manifest_path, {**manifest, "fingerprint": fingerprint, "created_at": utc_now()})
    expected = len(questions) * samples
    with FileLock(str(root / ".generation.lock"), timeout=config.get("bank_lock_timeout_seconds", 0)):
        remaining = [(q, i) for q in questions for i in range(samples) if not (output_dir / f"{q['question_id']}__s{i:03d}.json").exists()]
        if not remaining:
            log("vLLM generation already complete", split=split, responses=expected)
            return
        model_path = snapshot_download(config["model_id"], revision=config["model_revision"], cache_dir=str(Path(config["cache_dir"]).resolve()), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        generation_config = read_json(Path(model_path) / "generation_config.json")
        stop_ids = generation_config["eos_token_id"]
        stop_ids = [stop_ids] if isinstance(stop_ids, int) else stop_ids
        prompt_ids = {q["question_id"]: tokenizer.apply_chat_template([{"role": "user", "content": q["problem"] + "\n\n" + config["prompt_suffix"]}], tokenize=True, add_generation_prompt=True) for q in questions}
        log("Initializing vLLM", split=split, pending=len(remaining))
        started = time.perf_counter()
        llm = LLM(model=model_path, tokenizer=model_path, dtype=config["dtype"], tensor_parallel_size=1, gpu_memory_utilization=config["gpu_memory_utilization"], max_model_len=config["max_model_len"], max_num_seqs=config["max_num_seqs"], max_num_batched_tokens=config["max_num_batched_tokens"], enable_prefix_caching=True, limit_mm_per_prompt={"image": 0}, generation_config="vllm", logprobs_mode="raw_logprobs", max_logprobs=1, seed=config["seed"])
        init_seconds = time.perf_counter() - started
        elapsed_generation, new_tokens, new_count = 0.0, 0, 0
        completed = expected - len(remaining)
        for offset in range(0, len(remaining), config["request_chunk_size"]):
            chunk = remaining[offset:offset + config["request_chunk_size"]]
            prompts = [{"prompt_token_ids": prompt_ids[q["question_id"]]} for q, _ in chunk]
            params = [SamplingParams(n=1, temperature=config["temperature"], top_p=config["top_p"], top_k=config["top_k"], max_tokens=config["max_new_tokens"], seed=sample_seed(config["seed"], q["question_id"], i), logprobs=1, stop_token_ids=stop_ids, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0) for q, i in chunk]
            call_started = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            call_seconds = time.perf_counter() - call_started
            elapsed_generation += call_seconds
            for (question, sample_index), output, param in zip(chunk, outputs, params):
                sample = output.outputs[0]
                ids = list(sample.token_ids)
                if output.prompt_token_ids != prompt_ids[question["question_id"]]:
                    raise AssertionError("vLLM altered the supplied prompt token IDs")
                if not ids or sample.logprobs is None or len(ids) != len(sample.logprobs):
                    raise AssertionError("Missing or unaligned sampled-token logprobs")
                sampled_lp = [float(p[token].logprob) for token, p in zip(ids, sample.logprobs)]
                response_id = f"{question['question_id']}__s{sample_index:03d}"
                box = boxed_span(sample.text)
                record = {"response_id": response_id, "question_id": question["question_id"], "sample_index": sample_index, "fingerprint": fingerprint, "backend": "vllm", "seed": param.seed, "text": sample.text, "answer": box["answer"] if box else None, "token_ids": ids, "prompt_token_ids": output.prompt_token_ids, "vllm_raw_logprobs": sampled_lp, "generated_token_count": len(ids), "finish_reason": sample.finish_reason, "stop_reason": sample.stop_reason, "truncated": sample.finish_reason == "length", "extraction_failed": not bool(box and box["answer"]), "batch_generation_seconds": call_seconds, "created_at": utc_now()}
                write_json(output_dir / f"{response_id}.json", record)
                new_tokens += len(ids)
                new_count += 1
                completed += 1
            status = {"state": "running", "backend": "vllm", "split": split, "completed": completed, "expected": expected, "initialization_seconds": init_seconds, "generation_seconds_this_process": elapsed_generation, "new_responses": new_count, "new_tokens": new_tokens, "tokens_per_second": new_tokens / max(elapsed_generation, 1e-9), "updated_at": utc_now()}
            write_json(root / "generation_status.json", status)
            log("vLLM chunk saved", **status)
        write_json(root / "generation_status.json", {**status, "state": "complete"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.json")
    p.add_argument("--split", choices=["train", "dev", "calibration_a", "calibration_b", "test"], required=True)
    args = p.parse_args()
    generate(runtime_config(args.config), args.split)


if __name__ == "__main__":
    main()
