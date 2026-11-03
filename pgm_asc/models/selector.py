"""Positive-support PGM selector and batched forward prediction."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pgm_asc.data.banks import INPUT_KEYS


def batch_inputs(data, indices, n, with_embeddings=False):
    batch = {k: data[k][indices] for k in INPUT_KEYS}
    if with_embeddings:
        questions, inverse = torch.unique(data["question_index"][indices], return_inverse=True)
        batch["unique_h"] = data["embeddings"][questions]
        batch["unique_x"] = data["x"][questions * n + n - 1]
        batch["response_owner"] = inverse
    return batch


def predict(model, data):
    model.eval()
    outputs, weights = [], []
    with torch.no_grad():
        for start in range(0, len(data["target"]), 512):
            batch = batch_inputs(data, slice(start, start + 512), model.n, model.embedding_dim > 0)
            values, diagnostic = model(batch, diagnostics=True)
            outputs.append(values.cpu().numpy())
            weights.append(diagnostic["weights"].cpu().numpy())
    return np.concatenate(outputs), np.concatenate(weights)


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

    @property
    def alpha(self):
        return self.beta.sigmoid() if self.fixed_alpha is None else self.beta * 0 + self.fixed_alpha

    def forward(self, batch, diagnostics=False):
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
                # Encode each response once per question, then select its prefix.
                # No pooling over future responses occurs before applying masks.
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
        # A finite sentinel prevents NaN gradients through logsumexp on columns
        # that are absent in every partition (all -inf would have undefined grad).
        logits = logits.masked_fill(~batch["candidate_mask"][:, None, :], -1e9)
        conditional = logits.log_softmax(-1)
        if self.mode == "single":
            logp = conditional[:, self.partition]
            weights = torch.zeros(sizes.shape[:2], device=x.device)
            weights[:, self.partition] = 1
        else:
            gate = self.gate(gate_features).squeeze(-1)
            if self.mode in ("uniform", "candidate_mlp", "deepsets"):
                gate = gate * 0
            logweights = gate.masked_fill(~pmask, -torch.inf).log_softmax(-1)
            logp = torch.logsumexp(logweights[:, :, None] + conditional, dim=1)
            weights = logweights.exp()
        if diagnostics:
            return logp, {"alpha": float(self.alpha.detach()), "weights": weights.detach()}
        return logp
