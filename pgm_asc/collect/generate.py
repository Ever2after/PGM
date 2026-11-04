"""Stage 1 (GPU): sample 16 responses per question with vLLM, then run the two answer-extraction passes.

Each question is written to the workspace once complete, so a stopped job resumes where it left off.
Sampling uses the model's own generation defaults (checked against configs/models.json) and a
deterministic per-request seed.
"""
from __future__ import annotations

import copy
import importlib.metadata
import time

from filelock import FileLock

from pgm_asc.collect.extraction import confidence_groups, prediction, self_certainty
from pgm_asc.collect.workspace import Workspace, check_test_allowed
from pgm_asc.config import MODELS, STUDY
from pgm_asc.data.bank import N_SAMPLES, load_cohort
from pgm_asc.utils import digest, log, read_json, sample_seed, utc_now, write_json

MAX_MODEL_LEN = 8192
TOP_LOGPROBS = 20


def request_seed(qid, index, phase):
    return sample_seed(0, f"{qid}/{phase}", index) % (2**31 - 1)


def render(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extraction_prompt(question, response, dataset):
    instruction = ("Now, output only the numerical integer value of the answer. Final answer: " if dataset == "gsm8k"
                   else r"Now, output ONLY the answer in the form of $\boxed{{}}$. Therefore, the answer is")
    return [{"role": "user", "content": question}, {"role": "assistant", "content": response}, {"role": "user", "content": instruction}]


def engine(model_id, revision, max_num_seqs, memory, cache_dir=None, max_model_len=MAX_MODEL_LEN):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM
    path = snapshot_download(model_id, revision=revision, cache_dir=cache_dir)
    llm = LLM(model=path, tensor_parallel_size=1, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=memory, max_num_seqs=max_num_seqs, max_num_batched_tokens=8192,
              enforce_eager=False, enable_prefix_caching=True, seed=0,
              compilation_config={"level": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                  "cudagraph_capture_sizes": [x for x in [1, 2, 4, 8, 16, 32, 64, 96, 128] if x <= max_num_seqs]},
              **({"limit_mm_per_prompt": {"image": 0}} if "gemma-3" in model_id else {}))
    return llm, AutoTokenizer.from_pretrained(path)


def make_record(q, index, output, prompt, tok, seconds, identity):
    seq = output.outputs[0]
    ids = list(seq.token_ids)
    if len(ids) != len(seq.logprobs):
        raise ValueError("Token/logprob length mismatch")
    scores, sampled = [], []
    for token, candidates in zip(ids, seq.logprobs):
        if token not in candidates:
            raise ValueError("Missing sampled-token logprob")
        scores.append(self_certainty([v.logprob for v in candidates.values()]))
        sampled.append(float(candidates[token].logprob))
    c = confidence_groups(scores)
    return {"question_id": q["question_id"], "response_id": f"{q['question_id']}__s{index:03d}", "sample_index": index,
            "text": seq.text, "token_ids": ids, "prompt_token_ids": tok.encode(prompt, add_special_tokens=False),
            "native_sample_logprobs": sampled, "self_certainty_per_token": scores,
            "lowest_confidence": c["lowest"], "bottom10_confidence": c["bottom10"],
            "generated_token_count": len(ids), "truncated": seq.finish_reason == "length", "finish_reason": seq.finish_reason,
            "native_generation_seconds_share": seconds, "generation_identity": identity,
            "request_seed": request_seed(q["question_id"], index, "body")}


def generate(model, dataset, split, shard=0, shards=2, limit=None, memory=0.25, cache_dir=None, allow_test=False):
    spec = MODELS[model]
    max_new = STUDY["datasets"][dataset]["max_new_tokens"]
    ws = Workspace(model, dataset, split)
    check_test_allowed(model, dataset, split, allow_test)
    qs = load_cohort(dataset, split)[:limit]
    qs = [q for i, q in enumerate(qs) if i % shards == shard]
    omni = dataset == "omnimath"
    with FileLock(str(ws.root / f".generation_shard{shard}.lock"), timeout=0):
        pending = [q for q in qs if not ws.done("questions", q["question_id"])]
        if not pending:
            return
        llm, tok = engine(spec["model_id"], spec["revision"], 64 if omni else 128, memory, cache_dir)
        params = llm.get_default_sampling_params()
        for key, value in spec["sampling"].items():
            if getattr(params, key) != value:
                raise ValueError(f"Model default {key}={getattr(params, key)} differs from configs/models.json ({value})")
        params.max_tokens, params.logprobs, params.prompt_logprobs = max_new, TOP_LOGPROBS, 0
        ep = llm.get_default_sampling_params()
        ep.max_tokens = 128
        definition = {"model_id": spec["model_id"], "revision": spec["revision"], "sampling": spec["sampling"], "max_tokens": max_new,
                      "dataset": dataset, "extraction_max_tokens": 128, "logprobs": TOP_LOGPROBS, "vllm": importlib.metadata.version("vllm"),
                      "shards": shards, "request_seed": "hash(seed0,question_id,sample_index,body/inline/final) mod (2**31-1)"}
        identity = digest(definition)
        manifest = ws.root / "generation_manifests" / f"shard{shard}.json"
        if manifest.exists() and read_json(manifest)["identity"] != identity:
            raise ValueError("Generation settings changed since this workspace was started")
        if not manifest.exists():
            write_json(manifest, definition | {"identity": identity, "shard": shard, "created_at": utc_now()})

        def sample(requests):
            if not requests:
                return []
            prompts = [render(tok, [{"role": "user", "content": q["problem"]}]) for q, _ in requests]
            if any(len(tok.encode(p, add_special_tokens=False)) + max_new > MAX_MODEL_LEN for p in prompts):
                raise ValueError("Context exceeded")
            sampling = []
            for q, i in requests:
                p = copy.deepcopy(params)
                p.seed = request_seed(q["question_id"], i, "body")
                sampling.append(p)
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - start
            return [make_record(q, i, o, p, tok, elapsed / len(requests), identity) for (q, i), o, p in zip(requests, outputs, prompts)]

        def extract(pairs, phase):
            if not pairs:
                return
            prompts = [render(tok, extraction_prompt(q["problem"], r["text"], dataset)) for q, r in pairs]
            sampling = []
            for q, r in pairs:
                p = copy.deepcopy(ep)
                p.seed = request_seed(q["question_id"], r["sample_index"], phase)
                sampling.append(p)
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - start
            for (q, r), p, o in zip(pairs, prompts, outputs):
                r["extraction_" + phase] = {"text": o.outputs[0].text, "input_tokens": len(tok.encode(p, add_special_tokens=False)),
                                            "output_tokens": len(o.outputs[0].token_ids), "seconds_share": elapsed / len(pairs),
                                            "request_seed": request_seed(q["question_id"], r["sample_index"], phase)}

        # Calibration first responses are sampled across questions (as in the ReASC calibration runner).
        if split == "calibration":
            first = [q for q in qs if not ws.done("first", q["question_id"])]
            for start in range(0, len(first), 128):
                chunk = first[start:start + 128]
                for q, r in zip(chunk, sample([(q, 0) for q in chunk])):
                    ws.write("first", q["question_id"], r)
        step = 4 if omni else 8
        for offset in range(0, len(pending), step):
            batch = pending[offset:offset + step]
            group, requests, new = {}, [], []
            for q in batch:
                qid = q["question_id"]
                if ws.done("staged", qid):
                    group[qid] = ws.read("staged", qid)["records"]
                    continue
                group[qid] = [ws.read("first", qid)] if split == "calibration" else []
                requests.extend((q, i) for i in range(1 if split == "calibration" else 0, N_SAMPLES))
                new.append(q)
            for r in sample(requests):
                group[r["question_id"]].append(r)
            extract([(q, r) for q in new for r in group[q["question_id"]]], "inline")
            for q in new:
                ws.write("staged", q["question_id"], {"records": group[q["question_id"]]})
            extract([(q, r) for q in batch for r in group[q["question_id"]]], "final")
            for q in batch:
                records = group[q["question_id"]]
                if [r["sample_index"] for r in records] != list(range(N_SAMPLES)):
                    raise ValueError("Incomplete question/sample order")
                for r in records:
                    r["pred"] = r["extraction_final"]["text"].strip()
                    r["prediction"] = prediction(r["text"], r["pred"], dataset)
                    chat = [{"role": "user", "content": q["problem"]}, {"role": "assistant", "content": r["text"]}]
                    r["reference_chat_tokens"] = len(tok.encode(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False),
                                                                 add_special_tokens=False))
                    ws.write("responses", r["response_id"], r)
                ws.write("questions", q["question_id"], {"question_id": q["question_id"], "responses": N_SAMPLES,
                                                          "generation_identity": identity, "completed_at": utc_now()})
            log("generated", model=model, dataset=dataset, split=split, shard=shard,
                complete=sum(ws.done("questions", q["question_id"]) for q in qs), total=len(qs))
