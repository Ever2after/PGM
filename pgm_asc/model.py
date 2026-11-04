"""The PGM aggregation head and its matched structural controls.

For a prefix of t responses with answer candidates a, partition m gives

    l_m(a) = sum_{j<=t, a_j=a} |C_m(j)|^(alpha-1) * softplus(f_sup(x_j)),   l(OTHER) = f_oth(w)
    p_m    = softmax(l_m over observed candidates and OTHER)
    p(a)   = sum_m softmax_m(f_gate(g_m)) * p_m(a)

``ARMS`` lists the mixture and every ablation; all share inputs, optimizer and labels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from pgm_asc.data.features import INPUT_KEYS

CAPACITY = 32

ARMS = {
    "mixture": {"mode": "mixture"},
    "uniform": {"mode": "uniform"},                                 # gate fixed to uniform
    "random": {"mode": "random"},                                   # size-matched random partitions
    "alpha1": {"mode": "mixture", "alpha": 1.},                     # no cluster-size discount
    "single_partition": {"mode": "single", "partition": 1},         # cosine partition at the 10% threshold
    "all_one": {"mode": "single", "partition": 0},
    "singletons": {"mode": "single", "partition": 5},
    "candidate_counts": {"mode": "candidate_mlp", "count_only": True},
    "candidate_confidence": {"mode": "candidate_mlp"},
    "deepsets_confidence": {"mode": "deepsets"},
    "deepsets_embedding": {"mode": "deepsets", "use_embeddings": True},
}
# Evaluated without retraining: the mixture checkpoint restricted to partition 1.
SAME_HEAD_SINGLE = {"mode": "single", "partition": 1}


def head(inputs, outputs=1):
    return nn.Sequential(nn.Linear(inputs, 32), nn.GELU(), nn.Linear(32, outputs))


class PGM(nn.Module):
    def __init__(self, n, mode="mixture", alpha=None, partition=3, embedding_dim=0):
        super().__init__()
        self.n, self.mode, self.fixed_alpha, self.partition = n, mode, alpha, partition
        self.support = head(9)
        self.gate = head(12)
        self.other = head(25)
        self.beta = nn.Parameter(torch.tensor(0.))
        self.candidate = head(6)
        self.embedding_dim = embedding_dim
        self.encoder = head(9 + embedding_dim, 16)
        self.deepset = head(32)
        self.count_only = False

    @property
    def alpha(self):
        return self.beta.sigmoid() if self.fixed_alpha is None else self.beta * 0 + self.fixed_alpha

    def forward(self, batch):
        x, ids = batch["x"], batch["ids"]
        response_mask = batch["response_mask"]
        sizes = batch["random_sizes"] if self.mode == "random" else batch["sizes"]
        gate_features = batch["random_gate"] if self.mode == "random" else batch["gate"]
        pmask = batch["random_partition_mask"] if self.mode == "random" else batch["partition_mask"]
        score = F.softplus(self.support(x).squeeze(-1)) * response_mask
        support_mask = (ids >= 0) & response_mask
        contributions = sizes.pow(self.alpha - 1) * score[:, None, :] * support_mask[:, None, :]
        logits = torch.zeros((*sizes.shape[:2], self.n + 1), device=x.device)
        index = ids.clamp_min(0)[:, None, :].expand_as(contributions)
        logits.scatter_add_(2, index, contributions)
        other = self.other(batch["w"]).squeeze(-1)
        logits[:, :, self.n] = other[:, None]
        if self.mode == "candidate_mlp":
            candidate_logits = self.candidate(batch["candidate"]).squeeze(-1)
            logits = candidate_logits[:, None, :].expand(-1, sizes.shape[1], -1).clone()
            logits[:, :, self.n] = other[:, None]
        elif self.mode == "deepsets":
            if self.embedding_dim:
                # Encode each response once per question; masks select the causal prefix.
                unique = self.encoder(torch.cat([batch["unique_x"], batch["unique_h"]], -1))
                encoded = unique[batch["response_owner"]] * support_mask[:, :, None]
            else:
                encoded = self.encoder(x) * support_mask[:, :, None]
            local = torch.zeros((len(x), self.n + 1, 16), device=x.device)
            local.scatter_add_(1, ids.clamp_min(0)[:, :, None].expand(-1, -1, 16), encoded)
            global_sum = encoded.sum(1)[:, None, :].expand_as(local)
            values = self.deepset(torch.cat([local, global_sum], -1)).squeeze(-1)
            logits = values[:, None, :].expand(-1, sizes.shape[1], -1).clone()
            logits[:, :, self.n] = other[:, None]
        # A finite sentinel keeps logsumexp gradients defined for absent candidates.
        logits = logits.masked_fill(~batch["candidate_mask"][:, None, :], -1e9)
        conditional = logits.log_softmax(-1)
        if self.mode == "single":
            return conditional[:, self.partition]
        gate = self.gate(gate_features).squeeze(-1)
        if self.mode in ("uniform", "candidate_mlp", "deepsets"):
            gate = gate * 0
        logweights = gate.masked_fill(~pmask, -torch.inf).log_softmax(-1)
        return torch.logsumexp(logweights[:, :, None] + conditional, dim=1)


def make(spec, embedding_dim):
    params = {k: v for k, v in spec.items() if k not in ("count_only", "use_embeddings")}
    if spec.get("use_embeddings"):
        params["embedding_dim"] = embedding_dim
    model = PGM(CAPACITY, **params)
    model.count_only = spec.get("count_only", False)
    return model


def batch(data, idx, model):
    b = {k: data[k][idx] for k in INPUT_KEYS}
    if model.count_only:
        b["candidate"] = b["candidate"].clone()
        b["candidate"][:, :, 2:] = 0
        b["w"] = b["w"].clone()
        b["w"][:, 8:] = 0
    if model.embedding_dim:
        q, inverse = torch.unique(data["question_index"][idx], return_inverse=True)
        b.update(unique_h=data["embeddings"][q], unique_x=data["x"][q * 16 + 15], response_owner=inverse)
    return b


def predict(model, data):
    """Log-probabilities over 32 candidate slots + OTHER for every prefix row."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(data["t"]), 512):
            out.append(model(batch(data, slice(start, start + 512), model)).numpy())
    return np.concatenate(out)


def restore(entry):
    """Rebuild a head from a checkpoint entry {state_dict, spec, embedding_dim}."""
    model = make(entry["spec"], entry["embedding_dim"])
    model.load_state_dict(entry["state_dict"])
    model.eval()
    return model
