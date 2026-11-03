"""Frozen response-bank loading and train-only prefix packing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pgm_asc.core.answers import answer_groups, equivalent
from pgm_asc.core.io import digest, log, read_json, sample_seed
from pgm_asc.data.features import prefix_features
from pgm_asc.models.partitions import OnlinePartitions, unit


def load_banks(questions, bank_root, n):
    banks = []
    for q in questions:
        root = Path(bank_root) / q.get("bank_split", "train")
        records, vectors = [], []
        for i in range(n):
            name = f"{q['question_id']}__s{i:03d}"
            record = read_json(root / "scored_responses" / f"{name}.json")
            if record["question_id"] != q["question_id"] or record["sample_index"] != i:
                raise ValueError("Response identity mismatch")
            with np.load(root / "embeddings" / f"{name}.npz", allow_pickle=False) as saved:
                vectors.append(saved["final_full"].copy())
            records.append(record)
        banks.append({"question": q, "records": records, "vectors": np.stack(vectors)})
        if len(banks) % 50 == 0:
            log("PGM response bank loaded", questions=len(banks), total=len(questions), n=n)
    return banks


def fit_representation(banks, centered=False):
    mean = np.mean([unit(b["vectors"]).mean(0) for b in banks], axis=0) if centered else np.zeros(banks[0]["vectors"].shape[1])
    sampled = []
    pair_ids = []
    for bank in banks:
        raw = bank["vectors"]
        valid = np.linalg.norm(raw, axis=1) > 1e-8
        vectors = unit(unit(raw) - mean)
        rows, cols = np.triu_indices(len(vectors), 1)
        eligible = np.flatnonzero(valid[rows] & valid[cols])
        if len(eligible):
            rng = np.random.default_rng(sample_seed(42, bank["question"]["question_id"], 0))
            selected = rng.choice(eligible, min(128, len(eligible)), replace=False)
            sampled.extend((vectors[rows[selected]] * vectors[cols[selected]]).sum(1).tolist())
            pair_ids.append({"question_id": bank["question"]["question_id"], "pairs": np.stack([rows[selected], cols[selected]], -1).tolist()})
    thresholds = np.quantile(sampled, [.1, .4, .7, .9]).tolist()
    return {"centered": centered, "mean": mean.tolist(), "thresholds": thresholds,
            "pair_sampling_hash": digest(pair_ids), "pair_samples": sampled, "pair_ids": pair_ids,
            "train_question_hash": digest([b["question"]["question_id"] for b in banks])}


def response_x(records):
    return np.asarray([[r["reasoning_mean_logprob"], r["answer_mean_logprob"],
        r["reasoning_mean_entropy"], np.log1p(r["generated_token_count"]),
        np.log1p(r["reasoning_token_count"]), np.log1p(r["answer_token_count"]),
        float(r["truncated"]), float(r["reasoning_token_count"] == 0),
        float(r["answer_token_count"] == 0)] for r in records], dtype=np.float32)


def pack_history(records, vectors, representation, n, seed=42):
    """Only observed responses; no gold parameter, no future representations."""
    if len(records) > n:
        raise ValueError("History exceeds configured capacity")
    transformed = unit(unit(vectors) - np.asarray(representation["mean"], dtype=np.float32))
    transformed[np.linalg.norm(vectors, axis=1) <= 1e-8] = 0
    partitions = OnlinePartitions(representation["thresholds"], seed)
    representatives, groups = answer_groups([r["answer"] for r in records])
    x = response_x(records)
    rows = []
    for t in range(1, len(records) + 1):
        normal = partitions.update(transformed[t - 1])
        random = partitions.snapshot(random=True)
        current_ids = np.asarray(groups[:t], dtype=np.int64)
        candidate_count = max(current_ids, default=-1) + 1
        counts = np.bincount(current_ids[current_ids >= 0], minlength=candidate_count)
        f = prefix_features(records[:t], transformed[:t])
        w = np.concatenate([f["count"], f["confidence"]]).astype(np.float32)
        row = {"x": np.zeros((n, 9), dtype=np.float32),
               "ids": np.full(n, -1, dtype=np.int64),
               "response_mask": np.arange(n) < t,
               "candidate_mask": np.r_[np.arange(n) < candidate_count, True],
               "candidate": np.zeros((n + 1, 6), dtype=np.float32),
               "w": w, "t": t,
               "leader": int(counts.argmax()) if candidate_count else n,
               "token_cost": sum(r["generated_token_count"] for r in records[:t])}
        row["x"][:t] = x[:t]
        row["ids"][:t] = current_ids
        for name, snapshot in [("", normal), ("random_", random)]:
            row[name + "sizes"] = np.ones((6, n), dtype=np.float32)
            row[name + "sizes"][:, :t] = snapshot["sizes"]
            row[name + "gate"] = snapshot["gate"]
            row[name + "partition_mask"] = snapshot["partition_mask"]
        for k in range(candidate_count):
            support = x[:t][current_ids == k]
            row["candidate"][k] = [counts[k], counts[k] / t, *support[:, :3].mean(0), support[:, 3].mean()]
        row["representatives"] = representatives[:candidate_count]
        rows.append(row)
    return rows


INPUT_KEYS = ["x", "ids", "response_mask", "candidate_mask", "candidate", "w",
              "sizes", "gate", "partition_mask", "random_sizes", "random_gate", "random_partition_mask"]


def build_dataset(banks, representation, n, store_embeddings=False):
    arrays = {k: [] for k in INPUT_KEYS}
    metadata, targets, gold_groups = [], [], []
    for qi, bank in enumerate(banks):
        q = bank["question"]
        rows = pack_history(bank["records"], bank["vectors"], representation, n,
                            seed=sample_seed(42, q["question_id"], 1))
        # Gold is read only after inference inputs have been completed.
        for row in rows:
            matches = [k for k, answer in enumerate(row["representatives"]) if equivalent(q["gold_answer"], answer)]
            gold = matches[0] if matches else n
            targets.append(gold)
            gold_groups.append([k in matches for k in range(n)] + [gold == n])
            metadata.append({"question_id": q["question_id"], "question_index": qi,
                             "t": row["t"], "leader": row["leader"],
                             "token_cost": row["token_cost"], "representatives": row["representatives"]})
            for key in INPUT_KEYS:
                arrays[key].append(row[key])
        if (qi + 1) % 50 == 0:
            log("PGM prefix packing", questions=qi + 1, total=len(banks), n=n)
    tensors = {k: torch.from_numpy(np.stack(v)) for k, v in arrays.items()}
    tensors["target"] = torch.tensor(targets, dtype=torch.long)
    tensors["gold_groups"] = torch.tensor(gold_groups, dtype=torch.bool)
    tensors["question_index"] = torch.tensor([r["question_index"] for r in metadata], dtype=torch.long)
    tensors["t"] = torch.tensor([r["t"] for r in metadata], dtype=torch.long)
    if store_embeddings:
        mean = np.asarray(representation["mean"], dtype=np.float32)
        vectors = []
        for bank in banks:
            raw = bank["vectors"]
            transformed = unit(unit(raw) - mean)
            transformed[np.linalg.norm(raw, axis=1) <= 1e-8] = 0
            vectors.append(transformed)
        tensors["embeddings"] = torch.from_numpy(np.stack(vectors))
    return tensors, metadata


def fit_scalers(data):
    valid = data["response_mask"]
    scalers = {}
    for key, values in [("x", data["x"][valid]), ("w", data["w"]),
                        ("gate", data["gate"].flatten(0, 1)),
                        ("candidate", data["candidate"][data["candidate_mask"]])]:
        std = values.float().std(0, unbiased=False)
        # A train-constant feature must use unit scale; rare failures may appear
        # in dev/test even if their indicator was always zero in training.
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        scalers[key] = {"mean": values.float().mean(0).tolist(), "std": std.tolist()}
    scalers["version"] = "constant_features_unit_scale_v2"
    return scalers


def scale(data, scalers):
    result = dict(data)
    for key in ["x", "w", "gate", "random_gate", "candidate"]:
        source = "gate" if key == "random_gate" else key
        params = scalers[source]
        result[key] = (data[key] - torch.tensor(params["mean"])) / torch.tensor(params["std"])
    return result
