"""Prefix tensors for the aggregation heads.

For every question and every prefix t = 1..16 of its responses, ``build`` produces the
inputs the heads see (answer candidates, response features, partition sizes and gate
features, prefix statistics) plus the training/evaluation target. Inputs use only the
observed prefix; gold labels enter only ``target`` and ``gold_groups``.
"""
from __future__ import annotations

import numpy as np
import torch

from pgm_asc.data.bank import N_SAMPLES, exact_groups
from pgm_asc.data.partitions import N_PARTITIONS, OnlinePartitions, unit
from pgm_asc.utils import digest, sample_seed

INPUT_KEYS = ["x", "ids", "response_mask", "candidate_mask", "candidate", "w",
              "sizes", "gate", "partition_mask", "random_sizes", "random_gate", "random_partition_mask"]


def response_x(records):
    """Nine per-response features: log-probabilities, entropy, lengths and failure flags."""
    return np.asarray([[r["reasoning_mean_logprob"], r["answer_mean_logprob"],
        r["reasoning_mean_entropy"], np.log1p(r["generated_token_count"]),
        np.log1p(r["reasoning_token_count"]), np.log1p(r["answer_token_count"]),
        float(r["truncated"]), float(r["reasoning_token_count"] == 0),
        float(r["answer_token_count"] == 0)] for r in records], dtype=np.float32)


def mean_std(values):
    return [float(np.mean(values)), float(np.std(values))] if len(values) else [0.0, 0.0]


def output_tokens(record) -> int:
    """Generated body tokens plus both answer-extraction outputs."""
    return record["generated_token_count"] + record["extraction_inline_output_tokens"] + record["extraction_final_output_tokens"]


def fit_representation(banks):
    """Cosine thresholds for partitions 1-4: quantiles .1/.4/.7/.9 of sampled train response pairs."""
    mean = np.zeros(banks[0].vectors.shape[1])
    sampled, pair_ids = [], []
    for bank in banks:
        raw = bank.vectors
        valid = np.linalg.norm(raw, axis=1) > 1e-8
        vectors = unit(unit(raw) - mean)
        rows, cols = np.triu_indices(len(vectors), 1)
        eligible = np.flatnonzero(valid[rows] & valid[cols])
        if len(eligible):
            rng = np.random.default_rng(sample_seed(42, bank.question["question_id"], 0))
            selected = rng.choice(eligible, min(128, len(eligible)), replace=False)
            sampled.extend((vectors[rows[selected]] * vectors[cols[selected]]).sum(1).tolist())
            pair_ids.append({"question_id": bank.question["question_id"], "pairs": np.stack([rows[selected], cols[selected]], -1).tolist()})
    thresholds = np.quantile(sampled, [.1, .4, .7, .9]).tolist()
    return {"centered": False, "dimension": len(mean), "thresholds": thresholds,
            "pair_sampling_hash": digest(pair_ids), "train_question_hash": digest([b.question["question_id"] for b in banks])}


def representation_mean(representation):
    if "mean" in representation:
        return np.asarray(representation["mean"], dtype=np.float32)
    return np.zeros(representation["dimension"], dtype=np.float32)


def pack(records, vectors, representation, seed=42, capacity=32):
    """Prefix rows for one question. No labels enter this function."""
    if len(records) > capacity:
        raise ValueError("Capacity exceeded")
    transformed = unit(unit(vectors) - representation_mean(representation))
    transformed[np.linalg.norm(vectors, axis=1) <= 1e-8] = 0
    partitions = OnlinePartitions(representation["thresholds"], seed)
    reps, groups = exact_groups([r["prediction"] for r in records])
    x = response_x(records)
    rows = []
    for t in range(1, len(records) + 1):
        normal = partitions.update(transformed[t - 1])
        random = partitions.snapshot(random=True)
        ids = groups[:t]
        k = int(ids.max()) + 1
        counts = np.bincount(ids, minlength=k)
        leader = int(counts.argmax())
        ordered = sorted(counts, reverse=True) + [0, 0]
        p = counts / t
        # 25 prefix statistics: 8 vote statistics, then mean/std of 4 features over all and leader responses, truncation rate.
        w = [t, 1., k, ordered[0] / t, ordered[1] / t, (ordered[0] - ordered[1]) / t, float(-(p * np.log(p)).sum()), 0.]
        for mask in [np.ones(t, dtype=bool), ids == leader]:
            for col in range(4):
                w.extend(mean_std(x[:t, col][mask] if col < 3 else np.log1p([r["generated_token_count"] for r in records[:t]])[mask]))
        w.append(float(np.mean([r["truncated"] for r in records[:t]])))
        row = {"x": np.zeros((capacity, 9), np.float32), "ids": np.full(capacity, -1, np.int64),
               "response_mask": np.arange(capacity) < t, "candidate_mask": np.r_[np.arange(capacity) < k, True],
               "candidate": np.zeros((capacity + 1, 6), np.float32), "w": np.asarray(w, np.float32),
               "t": t, "leader": leader, "representatives": reps[:k],
               "token_cost": sum(r["reference_chat_tokens"] for r in records[:t])}
        row["x"][:t] = x[:t]
        row["ids"][:t] = ids
        for prefix, snapshot in [("", normal), ("random_", random)]:
            row[prefix + "sizes"] = np.ones((N_PARTITIONS, capacity), np.float32)
            row[prefix + "sizes"][:, :t] = snapshot["sizes"]
            row[prefix + "gate"] = snapshot["gate"]
            row[prefix + "partition_mask"] = snapshot["partition_mask"]
        for c in range(k):
            support = x[:t][ids == c]
            row["candidate"][c] = [counts[c], counts[c] / t, *support[:, :3].mean(0), support[:, 3].mean()]
        rows.append(row)
    return rows


def build(banks, representation, capacity=32):
    """Stack all prefixes of all questions. Target: first correct candidate, else OTHER (= capacity)."""
    arrays = {k: [] for k in INPUT_KEYS}
    meta, targets, gold = [], [], []
    for qi, b in enumerate(banks):
        rows = pack(b.records, b.vectors, representation, sample_seed(42, b.question["question_id"], 1), capacity)
        labels = b.labels["canonical_correct"]
        for row in rows:
            matches = [k for k in range(len(row["representatives"])) if labels[k]]
            target = matches[0] if matches else capacity
            targets.append(target)
            gold.append([k in matches for k in range(capacity)] + [not matches])
            meta.append({k: row[k] for k in ["t", "leader", "token_cost", "representatives"]}
                        | {"question_index": qi, "question_id": b.question["question_id"]})
            for key in INPUT_KEYS:
                arrays[key].append(row[key])
    tensors = {k: torch.from_numpy(np.stack(v)) for k, v in arrays.items()}
    tensors.update(target=torch.tensor(targets, dtype=torch.long), gold_groups=torch.tensor(gold, dtype=torch.bool),
                   question_index=torch.tensor([r["question_index"] for r in meta]), t=torch.tensor([r["t"] for r in meta]))
    return tensors, meta


def fit_scalers(data):
    valid = data["response_mask"]
    scalers = {}
    for key, values in [("x", data["x"][valid]), ("w", data["w"]),
                        ("gate", data["gate"].flatten(0, 1)),
                        ("candidate", data["candidate"][data["candidate_mask"]])]:
        std = values.float().std(0, unbiased=False)
        # Train-constant features use unit scale; rare failures may appear only in dev/test.
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        scalers[key] = {"mean": values.float().mean(0).tolist(), "std": std.tolist()}
    scalers["version"] = "constant_features_unit_scale_v2"
    return scalers


def scale(data, scalers):
    result = dict(data)
    for key in ["x", "w", "gate", "random_gate", "candidate"]:
        params = scalers["gate" if key == "random_gate" else key]
        result[key] = (data[key] - torch.tensor(params["mean"])) / torch.tensor(params["std"])
    return result


def attach(data, banks, meta, representation, capacity=32):
    """Add embeddings (for DeepSets) and replace prefix cost by consumed output tokens."""
    vectors = np.zeros((len(banks), capacity, banks[0].vectors.shape[-1]), np.float32)
    for qi, b in enumerate(banks):
        raw = b.vectors
        v = unit(unit(raw) - representation_mean(representation))
        v[np.linalg.norm(raw, axis=1) <= 1e-8] = 0
        vectors[qi, :N_SAMPLES] = v
        total = 0
        for t, r in enumerate(b.records):
            total += output_tokens(r)
            row = meta[qi * N_SAMPLES + t]
            if row["question_id"] != b.question["question_id"]:
                raise ValueError("Cohort order mismatch")
            row["reference_chat_tokens"] = row["token_cost"]
            row["token_cost"] = total
    data["embeddings"] = torch.from_numpy(vectors)
    return data, meta


def prepare(banks, representation, scalers):
    """Scaled prefix tensors and metadata ready for the heads."""
    data, meta = build(banks, representation)
    return attach(scale(data, scalers), banks, meta, representation)
