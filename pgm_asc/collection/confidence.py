"""Full-vocabulary confidence extraction for paper-based baselines."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pgm_asc.collection.teacher_forcing import packed_batches
from pgm_asc.core.config import runtime_config
from pgm_asc.core.io import digest, log, read_json, utc_now, write_json


def bottom_group(values, window=128):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return 0.
    width = min(window, len(values))
    cumulative = np.r_[0., np.cumsum(values)]
    groups = (cumulative[width:] - cumulative[:-width]) / width
    k = max(1, int(np.ceil(.1 * len(groups))))
    return float(np.partition(groups, k - 1)[:k].mean())


def token_certainty(logits):
    logits = logits.float()
    return torch.logsumexp(logits, -1) - logits.mean(-1)


@torch.inference_mode()
def score(model, tokenizer, records, chunk_size):
    device = next(model.parameters()).device
    lengths = [len(r["prompt_token_ids"]) + len(r["token_ids"]) for r in records]
    tokens = torch.full((len(records), max(lengths)), tokenizer.pad_token_id, device=device, dtype=torch.long)
    attention = torch.zeros_like(tokens)
    for i, r in enumerate(records):
        ids = r["prompt_token_ids"] + r["token_ids"]
        tokens[i, :len(ids)] = torch.tensor(ids, device=device)
        attention[i, :len(ids)] = 1
    output = model.get_decoder()(input_ids=tokens, attention_mask=attention, use_cache=False, return_dict=True)
    prediction = torch.cat([output.last_hidden_state[i, len(r["prompt_token_ids"]) - 1:lengths[i] - 1]
                            for i, r in enumerate(records)])
    sampled = torch.cat([torch.tensor(r["token_ids"], device=device) for r in records])
    certainty, raw_lp = [], []
    for start in range(0, len(sampled), chunk_size):
        end = start + chunk_size
        logits = model.get_output_embeddings()(prediction[start:end]).float()
        normalizer = torch.logsumexp(logits, -1)
        certainty.append((normalizer - logits.mean(-1)).cpu().numpy())
        selected = logits.gather(1, sampled[start:end, None]).squeeze(1) - normalizer
        raw_lp.append(selected.cpu().numpy())
    certainty, raw_lp = np.concatenate(certainty), np.concatenate(raw_lp)
    special = set(tokenizer.all_special_ids)
    cursor, results = 0, []
    for r in records:
        n = len(r["token_ids"])
        c, lp = certainty[cursor:cursor + n], raw_lp[cursor:cursor + n]
        cursor += n
        mask = np.array([token not in special for token in r["token_ids"]], dtype=bool)
        difference = float(np.abs(lp - np.asarray(r["vllm_raw_logprobs"])).mean())
        if not np.isfinite(c).all() or difference >= .05:
            raise ValueError("Supplemental logits failed numeric/backend alignment checks")
        # Confidence LNS is calculated from the stored original vLLM raw
        # logprobs; new logits are needed only for full-vocabulary self-certainty.
        original_lp = np.asarray(r["vllm_raw_logprobs"])[mask]
        values = c[mask]
        results.append({"response_id": r["response_id"], "question_id": r["question_id"],
            "sample_index": r["sample_index"], "source_generation_fingerprint": r["fingerprint"],
            "source_token_hash": digest(r["token_ids"]), "scored_token_count": len(values),
            "self_certainty_mean": float(values.mean()) if len(values) else 0.,
            "reasc_bottom10_group128": bottom_group(values),
            "lns_geom": float(np.exp(original_lp.mean())) if len(original_lp) else 0.,
            "lns_arith": float(np.exp(original_lp).mean()) if len(original_lp) else 0.,
            "raw_logprob_backend_mae": difference})
    return results


def run(config_path, splits):
    from filelock import FileLock
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    config = runtime_config(config_path)
    definition = {"implementation": "reasc_raw_full_vocab_v1", "model_id": config["model_id"],
                  "model_revision": config["model_revision"], "window": 128, "stride": 1,
                  "bottom_fraction": .1, "bottom_rounding": "ceil_min_one", "short_response": "one_full_group",
                  "token_scope": "entire_generated_response_excluding_special_tokens",
                  "sampling_unchanged": True, "original_pgm_features_unchanged": True}
    fingerprint = digest(definition)
    model_path = snapshot_download(config["model_id"], revision=config["model_revision"],
                                  cache_dir=str(Path(config["cache_dir"]).resolve()), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Gemma3ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16,
                                                          attn_implementation="sdpa").to("cuda").eval()
    for split in splits:
        root = Path(config["run_dir"]) / split
        out = root / "baseline_features"
        out.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root / ".baseline_features.lock"), timeout=0):
            manifest_path = out / "manifest.json"
            if manifest_path.exists() and read_json(manifest_path)["fingerprint"] != fingerprint:
                raise ValueError("Existing supplemental confidence definition differs")
            write_json(manifest_path, {**definition, "fingerprint": fingerprint, "created_at": utc_now()})
            paths = sorted((root / "responses").glob("*.json"))
            records = [read_json(p) for p in paths if not (out / p.name).exists()]
            completed, started = len(paths) - len(records), time.perf_counter()
            status = {"state": "running", "completed": completed, "expected": len(paths),
                      "split": split, "updated_at": utc_now()}
            write_json(out / "status.json", status)
            try:
                for batch in packed_batches(records, config["tf_batch_size"], config["tf_max_batch_tokens"]):
                    for row in score(model, tokenizer, batch, config["logit_chunk_size"]):
                        write_json(out / f"{row['response_id']}.json", {**row, "feature_fingerprint": fingerprint})
                        completed += 1
                    status.update(completed=completed, elapsed_seconds=time.perf_counter() - started, updated_at=utc_now())
                    write_json(out / "status.json", status)
                    if completed % 128 == 0:
                        log("Supplemental baseline confidence", **status)
                status.update(state="complete", updated_at=utc_now())
                write_json(out / "status.json", status)
            except BaseException as error:
                status.update(state="failed", error=str(error), updated_at=utc_now())
                write_json(out / "status.json", status)
                raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.json")
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "calibration_a", "calibration_b", "test"], required=True)
    args = p.parse_args()
    run(args.config, args.splits)


if __name__ == "__main__":
    main()
