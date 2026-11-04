"""Stage 3 (GPU): xVerify-3B-Ia correctness labels (greedy decoding, pinned prompt/template).

Each question gets one label per exact-answer candidate (``canonical_correct``, the primary metric)
and one per raw response (``raw_response_correct``, the ReASC notebook's reference metric).
Labels are stored apart from every policy input.
"""
from __future__ import annotations

import ast
import time
from pathlib import Path

from pgm_asc.collect.workspace import Workspace, check_test_allowed
from pgm_asc.config import ROOT, STUDY
from pgm_asc.data.bank import N_SAMPLES, exact_groups, load_cohort
from pgm_asc.utils import digest, file_sha256, log, utc_now, write_json

XVERIFY = ROOT / "third_party" / "xverify_official" / "src" / "xVerify" / "prompts"
MAX_LEN = 16384


def literal(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(name)


def verdict(text):
    value = text.strip().strip("[]").strip().lower()
    if value not in ["correct", "incorrect"]:
        raise ValueError("Invalid xVerify verdict: " + repr(text))
    return value == "correct"


def judge(model, dataset, split, follow=False, limit=None, memory=0.25, cache_dir=None, allow_test=False):
    from filelock import FileLock
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    spec = STUDY["judge"]
    ws = Workspace(model, dataset, split)
    check_test_allowed(model, dataset, split, allow_test)
    qs = load_cohort(dataset, split)[:limit]
    prompt_file, template_file = XVERIFY / "judge_prompt.py", XVERIFY / "base_template.py"
    if not prompt_file.exists():
        raise SystemExit("xVerify prompts missing: run `python scripts/setup_references.py` first.")
    prompt = literal(prompt_file, "PROMPT")
    template = literal(template_file, "BASE_TEMPLATE")["xVerify-3B-Ia"]
    identity = digest({"judge": spec, "prompt": file_sha256(prompt_file), "template": file_sha256(template_file), "add_special_tokens": True})
    llm = tok = None
    with FileLock(str(ws.root / ".judge.lock"), timeout=0):
        while True:
            pending = [q for q in qs if ws.done("questions", q["question_id"]) and not ws.done("judgments", q["question_id"])]
            if pending and llm is None:
                path = snapshot_download(spec["model_id"], revision=spec["revision"], cache_dir=cache_dir)
                llm = LLM(model=path, tensor_parallel_size=1, dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=memory,
                          max_num_seqs=64, max_num_batched_tokens=8192, enforce_eager=False, enable_prefix_caching=True, seed=0,
                          compilation_config={"level": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]})
                tok = AutoTokenizer.from_pretrained(path)
            for q in pending:
                qid = q["question_id"]
                records = [ws.read("responses", f"{qid}__s{i:03d}") for i in range(N_SAMPLES)]
                reps, ids = exact_groups([r["prediction"] for r in records])
                unique = list(dict.fromkeys([r["text"] for r in records] + reps))
                prompts = [{"prompt_token_ids": tok.encode(template.format(query=prompt.format(question=q["problem"], output=s, answer=q["gold_answer"])),
                                                           add_special_tokens=True)} for s in unique]
                if any(len(p["prompt_token_ids"]) + spec["max_tokens"] > MAX_LEN for p in prompts):
                    raise ValueError("Judge context exceeded")
                start = time.perf_counter()
                generations = llm.generate(prompts, SamplingParams(temperature=0., max_tokens=spec["max_tokens"]), use_tqdm=False)
                results = {s: {"correct": verdict(g.outputs[0].text), "verdict": g.outputs[0].text,
                               "input_tokens": len(p["prompt_token_ids"]), "output_tokens": len(g.outputs[0].token_ids)}
                           for s, p, g in zip(unique, prompts, generations)}
                ws.write("judgments", qid, {"question_id": qid, "judge_identity": identity, "representatives": reps,
                                            "canonical_correct": [results[s]["correct"] for s in reps],
                                            "raw_response_correct": [results[r["text"]]["correct"] for r in records],
                                            "elapsed_seconds": time.perf_counter() - start,
                                            "details": [{"output": s, **r} for s, r in results.items()]})
            completed = sum(ws.done("judgments", q["question_id"]) for q in qs)
            write_json(ws.root / "judge_status.json", {"questions": completed, "total": len(qs), "updated_at": utc_now()})
            if completed == len(qs) or not follow:
                log("judged", model=model, dataset=dataset, split=split, questions=completed, total=len(qs))
                return
            time.sleep(5)
