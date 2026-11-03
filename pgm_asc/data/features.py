"""Causal count and response-confidence feature construction."""
from __future__ import annotations

from collections import Counter

import numpy as np

from pgm_asc.core.answers import answer_groups


COUNT_NAMES = ["t", "valid_fraction", "unique_answers", "leader_fraction", "runner_fraction", "vote_margin", "answer_entropy", "extraction_failure_fraction"]


CONF_NAMES = [f"{subset}_{feature}_{stat}" for subset in ["all", "leader"] for feature in ["reasoning_mean_logprob", "answer_mean_logprob", "reasoning_mean_entropy", "log_length"] for stat in ["mean", "std"]] + ["truncation_fraction"]


REL_NAMES = ["all_pair_cosine_mean", "all_pair_cosine_std", "leader_pair_cosine_mean", "leader_pair_cosine_std", "leader_other_cosine_mean", "has_all_pair", "has_leader_pair", "has_leader_other_pair", "embedding_valid_fraction"]


def mean_std(values):
    return [float(np.mean(values)), float(np.std(values))] if len(values) else [0.0, 0.0]


def prefix_features(records, embeddings):
    """Features and leader from ONLY the supplied prefix; never receives gold."""
    t = len(records)
    representatives, groups = answer_groups([r["answer"] for r in records])
    groups = np.asarray(groups)
    counts = Counter(groups[groups >= 0].tolist())
    # Representatives are in first-observed order. A tie never consults gold.
    leader = max(sorted(counts), key=lambda k: counts[k]) if counts else -1
    ordered = sorted(counts.values(), reverse=True) + [0, 0]
    valid = int((groups >= 0).sum())
    p = np.array(list(counts.values()), dtype=float) / valid if valid else np.array([])
    entropy = float(-(p * np.log(p)).sum()) if len(p) else 0.0
    leader_mask = groups == leader if leader >= 0 else np.zeros(t, dtype=bool)
    count_features = [t, valid / t, len(counts), ordered[0] / t, ordered[1] / t, (ordered[0] - ordered[1]) / t, entropy, 1 - valid / t]
    conf = []
    for mask in [np.ones(t, dtype=bool), leader_mask]:
        for feature in ["reasoning_mean_logprob", "answer_mean_logprob", "reasoning_mean_entropy", "log_length"]:
            values = [np.log1p(r["generated_token_count"]) if feature == "log_length" else r[feature] for i, r in enumerate(records) if mask[i]]
            conf.extend(mean_std(values))
    conf.append(float(np.mean([r["truncated"] for r in records])))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    embedding_valid = norms > 1e-8
    normalized = embeddings / np.maximum(norms[:, None], 1e-8)
    similarity = np.clip(normalized @ normalized.T, -1, 1)
    pairs = np.triu(np.ones((t, t), dtype=bool), k=1)
    pairs &= embedding_valid[:, None] & embedding_valid[None, :]
    leader_pairs = pairs & leader_mask[:, None] & leader_mask[None, :]
    cross_pairs = pairs & (leader_mask[:, None] != leader_mask[None, :]) & (groups[:, None] >= 0) & (groups[None, :] >= 0)
    all_cos, leader_cos, cross_cos = similarity[pairs], similarity[leader_pairs], similarity[cross_pairs]
    relation = mean_std(all_cos) + mean_std(leader_cos) + [float(cross_cos.mean()) if len(cross_cos) else 0.0, float(bool(len(all_cos))), float(bool(len(leader_cos))), float(bool(len(cross_cos))), float(embedding_valid.mean())]
    all_mask = embedding_valid
    lead_mask = embedding_valid & leader_mask
    all_mean = normalized[all_mask].mean(axis=0) if all_mask.any() else np.zeros(embeddings.shape[1])
    lead_mean = normalized[lead_mask].mean(axis=0) if lead_mask.any() else np.zeros(embeddings.shape[1])
    return {"count": np.array(count_features), "confidence": np.array(conf), "relation": np.array(relation), "representation": np.concatenate([all_mean, lead_mean]), "leader_answer": representatives[leader] if leader >= 0 else None, "leader_cosine": float(leader_cos.mean()) if len(leader_cos) else None}
