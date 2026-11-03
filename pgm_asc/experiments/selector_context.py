"""Cached selector-stage data, baseline traces, and predictions."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from pgm_asc.core.integrity import load_tensor
from pgm_asc.core.io import digest, read_json, read_jsonl, write_json, write_jsonl
from pgm_asc.data.banks import build_dataset, load_banks, scale
from pgm_asc.models.selector import PGM, predict
from pgm_asc.policies.baselines import cges_logp, reasc_calibrate, traces
from pgm_asc.policies.calibration import bucket_calibrator, calibrate, global_calibrator
from pgm_asc.policies.stopping import distribution_traces


class Context:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["run_dir"])
        self.source = Path(config["source_run_dir"])
        self.choices = read_json(self.source / "initial_selection.json")["choices"]
        if any(self.choices[m]["representation"] != "final_full"
               for m in config["followup_methods"] if m in self.choices):
            raise ValueError("v2 fixed representation control expects final_full")
        self.source_cache = self.source / "cache"
        self.datasets, self.metadata, self.lp, self.banks, self.histories = {}, {}, {}, {}, {}
        self.cges_settings = read_json("configs/experiment.json")["baselines"]["cges"]
        self.reasc_settings = None

    def data(self, split):
        if split in self.datasets:
            return self.datasets[split], self.metadata[split]
        if split in ["train", "dev"]:
            data = load_tensor(self.source_cache / f"{split}.pt")
            metadata = read_jsonl(self.source_cache / f"{split}_metadata.jsonl")
        else:
            cache = self.root / "cache" / split
            if (cache / "data.pt").exists() and (cache / "metadata.jsonl").exists():
                data = load_tensor(cache / "data.pt")
                metadata = read_jsonl(cache / "metadata.jsonl")
            else:
                if split == "test" and not (self.root / "selection_lock.json").exists():
                    raise ValueError("Holdout labels cannot be loaded before policy lock")
                if not read_json(self.root / split / "bank_audit.json")["passed"]:
                    raise ValueError("Incomplete bank")
                questions = read_jsonl(Path(self.config["data_dir"]) / f"{split}.jsonl")
                gold = {r["question_id"]: r["gold_answer"] for r in read_jsonl(
                    Path(self.config["data_dir"]) / f"{split}_gold.jsonl")}
                for q in questions:
                    q.update(bank_split=split, gold_answer=gold[q["question_id"]])
                banks = load_banks(questions, self.root, 32)
                data, metadata = build_dataset(banks, read_json(self.source_cache / "representation.json"), 32)
                data = scale(data, read_json(self.source_cache / "scalers.json"))
                cache.mkdir(parents=True, exist_ok=True)
                torch.save(data, cache / "data.pt")
                write_jsonl(cache / "metadata.jsonl", metadata)
        data.pop("embeddings", None)
        if split not in ['train', 'dev']:
            questions = read_jsonl(Path(self.config['data_dir']) / f'{split}.jsonl')
            if [r['question_id'] for r in metadata[::32]] != [q['question_id'] for q in questions]:
                raise ValueError('Packed data order differs from the frozen cohort')
        self.datasets[split], self.metadata[split] = data, metadata
        return data, metadata

    def baseline_banks(self, split):
        if split in self.banks:
            return self.banks[split]
        data, metadata = self.data(split)
        bankroot = self.source if split in ["train", "dev"] else self.root
        cache = self.root / "cache" / f"{split}_baseline_banks.joblib"
        if cache.exists():
            self.banks[split] = joblib.load(cache)
            return self.banks[split]
        from transformers import AutoTokenizer
        snapshot = Path(self.config["cache_dir"]) / "models--google--gemma-3-4b-it/snapshots" / self.config["model_revision"]
        special = set(AutoTokenizer.from_pretrained(snapshot).all_special_ids)
        banks = []
        for row in metadata[::32]:
            qid = row["question_id"]
            answers, features = [], []
            for index in range(32):
                name = f"{qid}__s{index:03d}.json"
                record = read_json(bankroot / split / "scored_responses" / name)
                answers.append(record["answer"])
                feature_path = bankroot / split / "baseline_features" / name
                if feature_path.exists():
                    feature = read_json(feature_path)
                    if feature["source_token_hash"] != digest(record["token_ids"]):
                        raise ValueError("Baseline token alignment mismatch")
                else:
                    mask = np.array([token not in special for token in record["token_ids"]], dtype=bool)
                    lp = np.asarray(record["vllm_raw_logprobs"])[mask]
                    feature = {"lns_geom": float(np.exp(lp.mean())) if len(lp) else 0.,
                               "lns_arith": float(np.exp(lp).mean()) if len(lp) else 0.}
                features.append(feature)
            banks.append({"question_id": qid, "answers": answers, "features": features})
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(banks, cache)
        self.banks[split] = banks
        return banks

    def raw(self, method, split, variant="original"):
        key = (method, split, variant)
        if key in self.lp:
            return self.lp[key]
        data, metadata = self.data(split)
        out = self.root / "predictions" / variant / method
        path = out / f"{split}.npz"
        if path.exists():
            with np.load(path) as saved:
                lp = saved["logp"].copy()
        elif method == "cges":
            lp = cges_logp(traces(self.baseline_banks(split), "cges", self.cges_settings))
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, logp=lp)
        else:
            checkpoint = (Path(self.choices[method]["path"]) if variant == "original" else
                          self.root / "refits" / method) / "checkpoint.pt"
            saved = load_tensor(checkpoint)
            model = PGM(32, **saved["spec"])
            model.load_state_dict(saved["state_dict"])
            lp, weights = predict(model, data)
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, logp=lp, gate_weights=weights)
        self.lp[key] = lp
        if lp.shape != (len(metadata), 33) or not np.isfinite(lp).all():
            raise ValueError('Invalid or stale prediction cache')
        return lp

    def calibrator(self, method, variant, kind):
        if method in ["asc", "reasc"]:
            return None
        path = self.root / "calibrators" / variant / method / f"{kind}.json"
        if path.exists():
            return read_json(path)
        data, _ = self.data("calibration_a")
        lp = self.raw(method, "calibration_a", variant)
        result = global_calibrator(lp, data) if kind == "global" else bucket_calibrator(lp, data)
        write_json(path, result)
        return result

    def trace(self, method, split, selected):
        key = (method, split, selected["variant"], digest(selected["calibrator"]))
        if key in self.histories:
            return self.histories[key]
        data, _ = self.data(split)
        if method in ["asc", "reasc"]:
            if method == "reasc" and self.reasc_settings is None:
                a, _ = self.data("calibration_a")
                self.reasc_settings = reasc_calibrate(self.baseline_banks("calibration_a"), a, 32)
            history = traces(self.baseline_banks(split), method, self.reasc_settings if method == "reasc" else {})
        else:
            lp = calibrate(self.raw(method, split, selected["variant"]), data["t"].numpy(), selected["calibrator"])
            history = distribution_traces(lp, data)
        self.histories[key] = history
        return history
