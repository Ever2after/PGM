"""Stage 2 (GPU): teacher-forcing pass over the exact sampled tokens.

Per response: mean log-probability and entropy of the reasoning span, mean log-probability of the
boxed answer span (raw distribution, before temperature/top-p), and the mean last-layer hidden
state over reasoning tokens (L2-normalized) used by the partitions and DeepSets.
"""
from __future__ import annotations

import time

import numpy as np

from pgm_asc.collect.extraction import boxed_span
from pgm_asc.collect.workspace import Workspace, check_test_allowed
from pgm_asc.config import MODELS
from pgm_asc.data.bank import N_SAMPLES, load_cohort
from pgm_asc.utils import log, read_json, utc_now, write_json


def token_span(tokenizer, token_ids, char_start, char_end):
    """Map decoded character spans onto the sampled token sequence (boundary tokens belong to the span)."""
    def prefix_len(n):
        return len(tokenizer.decode(token_ids[:n], skip_special_tokens=True, clean_up_tokenization_spaces=False))

    def first_at_least(pos):
        lo, hi = 0, len(token_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if prefix_len(mid) >= pos:
                hi = mid
            else:
                lo = mid + 1
        return lo
    return max(0, first_at_least(char_start + 1) - 1), first_at_least(char_end)


def packed_batches(records, max_batch_size, max_tokens):
    batch, longest = [], 0
    for record in sorted(records, key=lambda r: len(r["prompt_token_ids"]) + len(r["token_ids"])):
        n = len(record["prompt_token_ids"]) + len(record["token_ids"])
        if batch and (len(batch) >= max_batch_size or max(longest, n) * (len(batch) + 1) > max_tokens):
            yield batch
            batch, longest = [], 0
        batch.append(record)
        longest = max(longest, n)
    if batch:
        yield batch


def score_batch(model, tokenizer, records, logit_chunk_size=128):
    import torch
    with torch.inference_mode():
        device = next(model.parameters()).device
        lengths = [len(r["prompt_token_ids"]) + len(r["token_ids"]) for r in records]
        tokens = torch.full((len(records), max(lengths)), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(tokens)
        for i, (record, length) in enumerate(zip(records, lengths)):
            tokens[i, :length] = torch.tensor(record["prompt_token_ids"] + record["token_ids"], device=device)
            attention[i, :length] = 1
        output = model.get_decoder()(input_ids=tokens, attention_mask=attention, use_cache=False, output_hidden_states=True, return_dict=True)
        # The state BEFORE each sampled token predicts it; hidden_states below are the states AFTER each token.
        prediction_states = torch.cat([output.last_hidden_state[i, len(r["prompt_token_ids"]) - 1:lengths[i] - 1] for i, r in enumerate(records)])
        sampled = torch.cat([torch.tensor(r["token_ids"], device=device) for r in records])
        all_lp, all_entropy = [], []
        for start in range(0, len(sampled), logit_chunk_size):
            lp = model.get_output_embeddings()(prediction_states[start:start + logit_chunk_size]).float().log_softmax(dim=-1)
            all_lp.append(lp.gather(-1, sampled[start:start + logit_chunk_size, None]).squeeze(-1).cpu())
            all_entropy.append((-(lp.exp() * lp).sum(dim=-1)).cpu())
        all_lp, all_entropy = torch.cat(all_lp).numpy(), torch.cat(all_entropy).numpy()
        special = set(tokenizer.all_special_ids)
        result, cursor = [], 0
        for i, record in enumerate(records):
            ids = record["token_ids"]
            n = len(ids)
            lp, entropy = all_lp[cursor:cursor + n], all_entropy[cursor:cursor + n]
            cursor += n
            text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            box = boxed_span(text)
            a, b = token_span(tokenizer, ids, 0, box["box_start"] if box else len(text))
            reason = np.array([a <= j < b and token not in special for j, token in enumerate(ids)])
            answer = np.zeros(n, dtype=bool)
            if box and box["answer"]:
                a, b = token_span(tokenizer, ids, box["body_start"], box["body_end"])
                answer[a:b] = True
            p = len(record["prompt_token_ids"])
            positions = np.flatnonzero(reason)
            hidden = output.hidden_states[-1][i, p:p + n]
            if len(positions):
                vector = hidden[torch.tensor(positions, device=device)].float().mean(dim=0).cpu().numpy()
                norm = np.linalg.norm(vector)
                vector = vector / norm if norm else vector
            else:
                vector = np.zeros(hidden.shape[-1], dtype=np.float32)
            sampled_lp = np.asarray(record["native_sample_logprobs"])
            result.append(({**record, "text": text, "body_answer": box["answer"] if box else None,
                            "extraction_failed": not bool(box and box["answer"]),
                            "reasoning_mean_logprob": float(lp[reason].mean()) if reason.any() else 0.0,
                            "answer_mean_logprob": float(lp[answer].mean()) if answer.any() else 0.0,
                            "reasoning_mean_entropy": float(entropy[reason].mean()) if reason.any() else 0.0,
                            "reasoning_token_count": int(reason.sum()), "answer_token_count": int(answer.sum()),
                            "raw_logprob_backend_mean_abs_difference": float(np.mean(np.abs(lp - sampled_lp))),
                            "embedding_valid": bool(reason.any())}, vector.astype(np.float32)))
        return result


def features(model, dataset, split, follow=False, limit=None, memory=0.25, cache_dir=None, allow_test=False):
    import torch
    from filelock import FileLock
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration
    spec = MODELS[model]
    ws = Workspace(model, dataset, split)
    check_test_allowed(model, dataset, split, allow_test)
    names = {f"{q['question_id']}__s{i:03d}" for q in load_cohort(dataset, split)[:limit] for i in range(N_SAMPLES)}
    torch.cuda.set_per_process_memory_fraction(memory)
    lm = tok = None
    with FileLock(str(ws.root / ".features.lock"), timeout=0):
        while True:
            pending = sorted((names & ws.names("responses")) - ws.names("scored"))
            records = [ws.read("responses", n) for n in pending]
            if records and lm is None:
                path = snapshot_download(spec["model_id"], revision=spec["revision"], cache_dir=cache_dir)
                tok = AutoTokenizer.from_pretrained(path)
                if tok.pad_token_id is None:
                    # Padding is right-sided and attention-masked, so the pad id never reaches a feature.
                    tok.pad_token = tok.eos_token
                cls = Gemma3ForConditionalGeneration if "gemma-3" in spec["model_id"] else AutoModelForCausalLM
                lm = cls.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
            for batch in packed_batches(records, 8, 8192):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = score_batch(lm, tok, batch)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                (ws.root / "embeddings").mkdir(exist_ok=True)
                for r, vector in output:
                    r["feature_seconds_share"] = elapsed / len(batch)
                    r["feature_input_tokens"] = len(r["prompt_token_ids"]) + len(r["token_ids"])
                    np.save(ws.root / "embeddings" / f"{r['response_id']}.npy", vector, allow_pickle=False)
                    ws.write("scored", r["response_id"], r)
            total = len(names & ws.names("scored"))
            write_json(ws.root / "feature_status.json", {"responses": total, "total": len(names), "updated_at": utc_now()})
            if total == len(names) or not follow:
                log("features", model=model, dataset=dataset, split=split, scored=total, total=len(names))
                return
            time.sleep(5)
