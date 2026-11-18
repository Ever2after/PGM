"""Compact v2 inputs: one .npz per (model, dataset), built once from the released bank (``pgm v2 prepare``).

Per split (calibration/train/dev/test) and question, in cohort order, 16 responses in sampling order:

* ``ids``       [Q,16] exact-string answer candidate (first-observed order, identical to the v1 grouping)
* ``valid``     [Q,16] the response has a non-empty extracted answer
* ``correct``   [Q,16] xVerify label of candidate slot k (padded False); ``raw_correct`` per response
* ``x``         [Q,16,3] bottom-10% and lowest windowed self-certainty, log(1 + generated tokens)
* ``tokens``    [Q,16] output tokens (generated body + both answer-extraction outputs), the cost unit of v1
* ``sim_text``  [Q,16,16] Jaccard similarity of word 3-gram sets of the reasoning (text before the last \\boxed)
* ``sim_hidden``[Q,16,16] cosine of the bank's mean last-layer hidden state, centered on the train mean
"""
from __future__ import annotations

import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pgm_asc.config import bank_path, embedding_path
from pgm_asc.data.bank import N_SAMPLES, exact_groups, load_cohort
from pgm_asc.utils import log
from pgm_asc.v2 import SLOTS, data_path

SPLITS = ("calibration", "train", "dev", "test")
X_NAMES = ["bottom10_confidence", "lowest_confidence", "log_generated_tokens"]
COLUMNS = ["split", "question_id", "sample_index", "prediction", "text", "bottom10_confidence", "lowest_confidence",
           "generated_token_count", "extraction_inline_output_tokens", "extraction_final_output_tokens", "correct", "raw_correct"]
_WORD = re.compile(r"\w+|[^\w\s]")


def reasoning(text: str) -> str:
    cut = text.rfind("\\boxed")
    return text[:cut] if cut >= 0 else text


def shingles(text: str, n: int = 3) -> set:
    words = _WORD.findall(reasoning(text).lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(sets) -> np.ndarray:
    n = len(sets)
    out = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets[i] & sets[j])
            union = len(sets[i]) + len(sets[j]) - inter
            out[i, j] = out[j, i] = inter / union if union else 0.
    return out


def _unit(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm > 1e-8, v / np.maximum(norm, 1e-12), 0.)


def prepare(model: str, dataset: str, hidden: bool = True) -> dict:
    table = pq.read_table(bank_path(model, dataset), columns=COLUMNS)
    split_col = table.column("split").to_numpy(zero_copy_only=False)
    vectors = np.load(embedding_path(model, dataset), mmap_mode="r") if hidden else None
    rows_of = {s: np.flatnonzero(split_col == s) for s in SPLITS}
    mean = None
    if hidden:
        train = rows_of["train"]
        acc = np.zeros(vectors.shape[1])
        for start in range(0, len(train), 4096):
            acc += _unit(np.asarray(vectors[train[start:start + 4096]], dtype=np.float64)).sum(0)
        mean = acc / len(train)
    out = {}
    for split in SPLITS:
        questions = load_cohort(dataset, split)
        rows = rows_of[split]
        q = len(questions)
        if len(rows) != N_SAMPLES * q:
            raise ValueError(f"{model}/{dataset}/{split}: {len(rows)} responses for {q} questions")
        sub = table.take(pa.array(rows)).to_pydict()
        qid = np.asarray(sub["question_id"], dtype=object).reshape(q, N_SAMPLES)
        if (qid != np.asarray([x["question_id"] for x in questions], dtype=object)[:, None]).any() or \
                (np.asarray(sub["sample_index"]).reshape(q, N_SAMPLES) != np.arange(N_SAMPLES)).any():
            raise ValueError(f"{model}/{dataset}/{split}: bank order does not match the cohort")
        pred = np.asarray(sub["prediction"], dtype=object).reshape(q, N_SAMPLES)
        resp_correct = np.asarray(sub["correct"], dtype=bool).reshape(q, N_SAMPLES)
        ids = np.zeros((q, N_SAMPLES), np.int8)
        correct = np.zeros((q, SLOTS), bool)
        for i in range(q):
            _, g = exact_groups(list(pred[i]))
            ids[i] = g
            for k in range(g.max() + 1):
                correct[i, k] = resp_correct[i, np.flatnonzero(g == k)[0]]
        gen = np.asarray(sub["generated_token_count"], np.float64).reshape(q, N_SAMPLES)
        x = np.stack([np.asarray(sub["bottom10_confidence"], np.float64).reshape(q, N_SAMPLES),
                      np.asarray(sub["lowest_confidence"], np.float64).reshape(q, N_SAMPLES), np.log1p(gen)], -1)
        tokens = gen + np.asarray(sub["extraction_inline_output_tokens"]).reshape(q, N_SAMPLES) \
            + np.asarray(sub["extraction_final_output_tokens"]).reshape(q, N_SAMPLES)
        texts = np.asarray(sub["text"], dtype=object).reshape(q, N_SAMPLES)
        sim_text = np.stack([jaccard([shingles(t) for t in texts[i]]) for i in range(q)])
        arrays = {"question_id": qid[:, 0].astype(str), "ids": ids,
                  "valid": np.vectorize(lambda s: bool(s and s.strip()))(pred).astype(bool),
                  "correct": correct, "raw_correct": np.asarray(sub["raw_correct"], bool).reshape(q, N_SAMPLES),
                  "x": x.astype(np.float32), "tokens": tokens.astype(np.int32), "sim_text": sim_text}
        if hidden:
            v = _unit(_unit(np.asarray(vectors[rows], dtype=np.float64)) - mean).reshape(q, N_SAMPLES, -1)
            sim = np.einsum("qid,qjd->qij", v, v).astype(np.float32)
            bad = np.linalg.norm(v, axis=-1) < .5
            sim[bad[:, :, None] | bad[:, None, :]] = np.nan
            arrays["sim_hidden"] = sim
        out.update({f"{split}.{k}": v for k, v in arrays.items()})
        log("v2-prepare", model=model, dataset=dataset, split=split, questions=q)
    path = data_path(model, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x_names=np.asarray(X_NAMES), **out)
    return out


def load(model: str, dataset: str) -> dict:
    """{split: {name: array}} from the prepared .npz."""
    with np.load(data_path(model, dataset), allow_pickle=False) as f:
        result = {s: {} for s in SPLITS}
        for key in f.files:
            if "." in key:
                split, name = key.split(".", 1)
                result[split][name] = f[key]
    return result
