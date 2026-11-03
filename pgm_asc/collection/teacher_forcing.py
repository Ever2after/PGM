"""Confidence and embedding extraction on exact generated tokens."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pgm_asc.collection.token_stats import raw_statistics, token_span
from pgm_asc.core.answers import boxed_span
from pgm_asc.core.config import runtime_config
from pgm_asc.core.io import digest, log, read_json, utc_now, write_json


def pool_variants(hidden_states, positions, middle_layer, early_fraction):
    """Pool token states only at supplied reasoning positions."""
    if not positions:
        size = hidden_states[-1].shape[-1]
        return {name: np.zeros(size, dtype=np.float32) for name in ["final_full", "final_early80", "middle_full", "middle_early80"]}
    full = torch.tensor(positions, device=hidden_states[-1].device)
    early = full[:max(1, int(len(positions) * early_fraction))]
    result = {}
    for prefix, h in [("final", hidden_states[-1]), ("middle", hidden_states[middle_layer])]:
        for suffix, indices in [("full", full), ("early80", early)]:
            vector = h[indices].float().mean(dim=0).cpu().numpy()
            norm = np.linalg.norm(vector)
            result[f"{prefix}_{suffix}"] = vector / norm if norm else vector
    return result


def packed_batches(records, max_batch_size, max_tokens):
    batch = []
    longest = 0
    for record in sorted(records, key=lambda r: len(r["prompt_token_ids"]) + len(r["token_ids"])):
        n = len(record["prompt_token_ids"]) + len(record["token_ids"])
        if batch and (len(batch) >= max_batch_size or max(longest, n) * (len(batch) + 1) > max_tokens):
            yield batch
            batch, longest = [], 0
        batch.append(record)
        longest = max(longest, n)
    if batch:
        yield batch


@torch.inference_mode()
def score_batch(model, tokenizer, records, config):
    device = next(model.parameters()).device
    lengths = [len(r["prompt_token_ids"]) + len(r["token_ids"]) for r in records]
    tokens = torch.full((len(records), max(lengths)), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(tokens)
    for i, (record, length) in enumerate(zip(records, lengths)):
        tokens[i, :length] = torch.tensor(record["prompt_token_ids"] + record["token_ids"], device=device)
        attention[i, :length] = 1
    output = model.get_decoder()(input_ids=tokens, attention_mask=attention, use_cache=False, output_hidden_states=True, return_dict=True)
    prediction_states = torch.cat([output.last_hidden_state[i, len(r["prompt_token_ids"])-1:lengths[i]-1] for i,r in enumerate(records)])
    sampled_tokens = torch.cat([torch.tensor(r["token_ids"], device=device) for r in records])
    all_lp, all_entropy = [], []
    for start in range(0, len(sampled_tokens), config["logit_chunk_size"]):
        end = start + config["logit_chunk_size"]
        logits = model.get_output_embeddings()(prediction_states[start:end])
        lp, entropy = raw_statistics(logits, sampled_tokens[start:end])
        all_lp.append(lp.cpu())
        all_entropy.append(entropy.cpu())
    all_lp = torch.cat(all_lp).numpy()
    all_entropy = torch.cat(all_entropy).numpy()
    result, cursor = [], 0
    special = set(tokenizer.all_special_ids)
    for i, record in enumerate(records):
        ids = record["token_ids"]
        n = len(ids)
        lp, entropy = all_lp[cursor:cursor+n], all_entropy[cursor:cursor+n]
        cursor += n
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        box = boxed_span(text)
        end = box["box_start"] if box else len(text)
        a, b = token_span(tokenizer, ids, 0, end)
        reason_mask = np.array([a <= j < b and token not in special for j,token in enumerate(ids)])
        answer_mask = np.zeros(n, dtype=bool)
        if box and box["answer"]:
            a, b = token_span(tokenizer, ids, box["body_start"], box["body_end"])
            answer_mask[a:b] = True
        p = len(record["prompt_token_ids"])
        # hidden_states contain state AFTER each sampled token; prediction states
        # above instead use the state immediately BEFORE each sampled token.
        hs = tuple(h[i, p:p+n] for h in output.hidden_states)
        vectors = pool_variants(hs, np.flatnonzero(reason_mask).tolist(), config["middle_layer"], config["early_fraction"])
        vllm_lp = np.asarray(record["vllm_raw_logprobs"])
        scored = {**record, "text": text, "answer": box["answer"] if box else None, "extraction_failed": not bool(box and box["answer"]), "reasoning_mean_logprob": float(lp[reason_mask].mean()) if reason_mask.any() else 0.0, "answer_mean_logprob": float(lp[answer_mask].mean()) if answer_mask.any() else 0.0, "reasoning_mean_entropy": float(entropy[reason_mask].mean()) if reason_mask.any() else 0.0, "reasoning_token_count": int(reason_mask.sum()), "answer_token_count": int(answer_mask.sum()), "raw_logprob_backend_mean_abs_difference": float(np.mean(np.abs(lp-vllm_lp))), "raw_logprob_backend_mean_signed_difference": float(np.mean(lp-vllm_lp)), "embedding_valid": bool(reason_mask.any())}
        result.append((scored, vectors))
    return result


def extract(config, split):
    from filelock import FileLock
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    root = Path(config["run_dir"]) / split
    scored_dir, vector_dir = root / "scored_responses", root / "embeddings"
    scored_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(exist_ok=True)
    generation_manifest = read_json(root / "generation_manifest.json")
    manifest = {"generation_fingerprint": generation_manifest["fingerprint"], "method": "batched_causal_teacher_forcing_v1", "middle_layer": config["middle_layer"], "early_fraction": config["early_fraction"], "precision": "bfloat16", "raw_distribution": "before_temperature_and_top_p", "torch_version": torch.__version__}
    fingerprint = digest(manifest)
    path = root / "extraction_manifest.json"
    if path.exists() and read_json(path)["fingerprint"] != fingerprint:
        raise ValueError("Existing extracted features have different definitions")
    write_json(path, {**manifest, "fingerprint": fingerprint})
    records = [read_json(p) for p in sorted((root / "responses").glob("*.json"))]
    expected = len(records)
    imports = read_json(root / "import_manifest.json")["responses"] if (root / "import_manifest.json").exists() else {}
    for record in records:
        if record["fingerprint"] != generation_manifest["fingerprint"]:
            provenance = imports.get(record["response_id"])
            if not provenance or digest(record) != provenance["record_hash"]:
                raise ValueError("Mixed generation fingerprints without verified import provenance")
    with FileLock(str(root / ".extraction.lock"), timeout=config.get("bank_lock_timeout_seconds", 0)):
        pending = [r for r in records if not (scored_dir / f"{r['response_id']}.json").exists()]
        if not pending:
            log("Feature extraction already complete", split=split, responses=expected)
            return
        model_path = snapshot_download(config["model_id"], revision=config["model_revision"], cache_dir=str(Path(config["cache_dir"]).resolve()), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = Gemma3ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        completed = expected - len(pending)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for batch in packed_batches(pending, config["tf_batch_size"], config["tf_max_batch_tokens"]):
            results = score_batch(model, tokenizer, batch, config)
            for scored, vectors in results:
                response_id = scored["response_id"]
                vector_path = vector_dir / f"{response_id}.npz"
                with vector_path.with_suffix(".npz.tmp").open("wb") as f:
                    np.savez_compressed(f, **vectors)
                vector_path.with_suffix(".npz.tmp").replace(vector_path)
                scored["extraction_fingerprint"] = fingerprint
                write_json(scored_dir / f"{response_id}.json", scored)
                completed += 1
            torch.cuda.synchronize()
            status = {"state": "running", "split": split, "completed": completed, "expected": expected, "elapsed_seconds_this_process": time.perf_counter()-started, "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30, "updated_at": utc_now()}
            write_json(root / "extraction_status.json", status)
            log("Teacher-forced batch saved", **status)
        write_json(root / "extraction_status.json", {**status, "state": "complete"})
        del model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--split", choices=["train", "dev", "calibration_a", "calibration_b", "test"], required=True)
    args = parser.parse_args()
    extract(runtime_config(args.config), args.split)


if __name__ == "__main__":
    main()
