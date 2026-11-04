"""Causal online partitions of the observed responses.

Six partitions are maintained as responses arrive:
  0 all-one, 1-4 cosine-centroid clustering at four similarity thresholds, 5 singletons.
A size-matched random control reassigns responses while keeping every prefix's cluster-size multiset.
"""
from __future__ import annotations

import numpy as np

N_PARTITIONS = 6


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


class OnlinePartitions:
    def __init__(self, thresholds, seed=42):
        self.thresholds = list(thresholds)
        self.assignments = [[] for _ in range(len(thresholds) + 2)]
        self.sums = [[] for _ in thresholds]
        self.vectors = []
        self.rng = np.random.default_rng(seed)
        self.random_assignments = [[] for _ in self.assignments]

    def update(self, vector):
        vector = unit(vector)
        index = len(self.vectors)
        self.vectors.append(vector)
        self.assignments[0].append(0)
        self.assignments[-1].append(index)
        for m, threshold in enumerate(self.thresholds):
            sums = self.sums[m]
            if sums and np.linalg.norm(vector) > 1e-8:
                similarities = unit(np.stack(sums)) @ vector
                k = int(np.argmax(similarities))
                if similarities[k] < threshold or np.linalg.norm(sums[k]) < 1e-8:
                    k = len(sums)
            else:
                k = len(sums)
            if k == len(sums):
                sums.append(vector.copy())
            else:
                sums[k] += vector
            self.assignments[m + 1].append(k)
        # Random control: same cluster-size multiset at every prefix.
        for original, random in zip(self.assignments, self.random_assignments):
            k = original[-1]
            old_size = original[:-1].count(k)
            if old_size == 0:
                random.append(max(random, default=-1) + 1)
            else:
                sizes = np.bincount(random)
                candidates = np.flatnonzero(sizes == old_size)
                random.append(int(self.rng.choice(candidates)))
        return self.snapshot()

    def snapshot(self, random=False):
        """Cluster sizes per response, 12 gate features and a duplicate-partition mask."""
        assignments = self.random_assignments if random else self.assignments
        vectors = np.stack(self.vectors)
        t = len(vectors)
        valid = np.linalg.norm(vectors, axis=1) > 1e-8
        similarity = np.clip(vectors @ vectors.T, -1, 1)
        sizes, features, unique, seen = [], [], [], set()
        for assignment in assignments:
            labels = np.asarray(assignment, dtype=np.int64)
            counts = np.bincount(labels)
            sizes.append(counts[labels])
            mapping = {}
            key = tuple(mapping.setdefault(int(k), len(mapping)) for k in labels)
            unique.append(key not in seen)
            seen.add(key)
            pairs = np.triu(np.ones((t, t), dtype=bool), 1)
            pairs &= valid[:, None] & valid[None, :]
            within = similarity[pairs & (labels[:, None] == labels[None, :])]
            centroids = [unit(vectors[(labels == k) & valid].sum(0))
                         for k in range(len(counts)) if ((labels == k) & valid).any()]
            if len(centroids) > 1:
                cm = np.stack(centroids) @ np.stack(centroids).T
                between = cm[np.triu_indices(len(cm), 1)]
            else:
                between = np.array([])
            proportions = counts / t
            features.append([
                len(counts) / t, float(-(proportions * np.log(proportions)).sum()),
                float(np.mean(counts == 1)), float(counts.max() / t),
                float(within.mean()) if len(within) else 0.,
                float(within.std()) if len(within) else 0.,
                float(between.mean()) if len(between) else 0.,
                float(between.std()) if len(between) else 0.,
                float(len(within) > 0), float(len(between) > 0),
                float(valid.mean()), float(t),
            ])
        return {"sizes": np.asarray(sizes, dtype=np.float32),
                "gate": np.asarray(features, dtype=np.float32),
                "partition_mask": np.asarray(unique, dtype=bool)}
