"""Cohorts and response banks.

* A cohort file holds the questions and gold answers of one dataset (all splits).
* A bank holds the 16 sampled responses per question for one (model, dataset), with
  per-response teacher-forcing features, answer-extraction costs and xVerify labels, plus
  an aligned float32 embedding matrix (one row per response).

Every method replays the same bank, so all comparisons share samples, order and labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pgm_asc.config import bank_path, cohort_path, embedding_path

N_SAMPLES = 16
SPLIT_ORDER = ("calibration", "train", "dev", "test")

LIST_I32 = pa.list_(pa.int32())
LIST_F32 = pa.list_(pa.float32())
BANK_SCHEMA = pa.schema([
    ("split", pa.string()), ("question_id", pa.string()), ("sample_index", pa.int16()),
    # Generation
    ("text", pa.string()), ("token_ids", LIST_I32), ("sampled_logprobs", LIST_F32), ("self_certainty", LIST_F32),
    ("finish_reason", pa.string()), ("truncated", pa.bool_()), ("generated_token_count", pa.int32()),
    ("reference_chat_tokens", pa.int32()), ("request_seed", pa.int64()), ("generation_identity", pa.string()),
    ("native_generation_seconds_share", pa.float64()),
    # ReASC self-certainty: windowed means over the per-token self-certainty above
    ("lowest_confidence", pa.float64()), ("bottom10_confidence", pa.float64()),
    # Two answer-extraction passes (inline, final) and the resulting prediction
    ("extraction_inline_input_tokens", pa.int32()), ("extraction_inline_output_tokens", pa.int32()),
    ("extraction_inline_seconds_share", pa.float64()),
    ("extraction_final_input_tokens", pa.int32()), ("extraction_final_output_tokens", pa.int32()),
    ("extraction_final_seconds_share", pa.float64()),
    ("pred", pa.string()), ("prediction", pa.string()),
    # Teacher-forcing features on the exact sampled tokens
    ("extraction_failed", pa.bool_()), ("embedding_valid", pa.bool_()),
    ("reasoning_mean_logprob", pa.float64()), ("answer_mean_logprob", pa.float64()), ("reasoning_mean_entropy", pa.float64()),
    ("reasoning_token_count", pa.int32()), ("answer_token_count", pa.int32()),
    ("feature_input_tokens", pa.int32()), ("feature_seconds_share", pa.float64()),
    # xVerify labels: the response's exact-answer candidate is correct / the raw response is correct
    ("correct", pa.bool_()), ("raw_correct", pa.bool_()),
])
# Columns needed to replay policies; text and token-level arrays are loaded only on request.
REPLAY_COLUMNS = [c for c in BANK_SCHEMA.names if c not in ("text", "token_ids", "sampled_logprobs", "self_certainty", "pred")]


@dataclass
class QuestionBank:
    question: dict
    records: list[dict]
    vectors: np.ndarray
    labels: dict = field(default_factory=dict)


def exact_groups(predictions):
    """Exact-string answer candidates in first-observed order (ReASC grouping)."""
    reps, ids = [], []
    for value in predictions:
        if value not in reps:
            reps.append(value)
        ids.append(reps.index(value))
    return reps, np.asarray(ids, dtype=np.int64)


def load_cohort(dataset: str, split: str) -> list[dict]:
    table = pq.read_table(cohort_path(dataset), filters=[("split", "=", split)])
    return [{k: v for k, v in row.items() if v is not None and k != "split"} for row in table.to_pylist()]


def load_bank(model: str, dataset: str, split: str, columns=None, embeddings=True) -> list[QuestionBank]:
    """Questions of one split in cohort order, each with its 16 responses."""
    questions = load_cohort(dataset, split)
    columns = REPLAY_COLUMNS if columns is None else list(dict.fromkeys(["split", "question_id", "sample_index", "prediction", "correct", "raw_correct", *columns]))
    file = pq.ParquetFile(bank_path(model, dataset))
    table = file.read(columns=columns)
    splits = table.column("split").to_numpy(zero_copy_only=False)
    rows = np.flatnonzero(splits == split)
    if len(rows) != N_SAMPLES * len(questions):
        raise ValueError(f"{model}/{dataset}/{split}: {len(rows)} responses for {len(questions)} questions")
    records = table.take(pa.array(rows)).to_pylist()
    vectors = np.load(embedding_path(model, dataset), mmap_mode="r") if embeddings else None
    banks = []
    for qi, question in enumerate(questions):
        chunk = records[qi * N_SAMPLES:(qi + 1) * N_SAMPLES]
        if any(r["question_id"] != question["question_id"] or r["sample_index"] != i for i, r in enumerate(chunk)):
            raise ValueError("Bank order does not match the cohort: " + question["question_id"])
        reps, ids = exact_groups([r["prediction"] for r in chunk])
        canonical = [bool(chunk[int(np.flatnonzero(ids == k)[0])]["correct"]) for k in range(len(reps))]
        labels = {"representatives": reps, "canonical_correct": canonical,
                  "raw_response_correct": [bool(r["raw_correct"]) for r in chunk]}
        start = rows[qi * N_SAMPLES]
        vec = np.array(vectors[start:start + N_SAMPLES], dtype=np.float32) if embeddings else None
        banks.append(QuestionBank(question, chunk, vec, labels))
    return banks


def write_bank(model: str, dataset: str, table: pa.Table, embeddings: np.ndarray, metadata: dict) -> None:
    """Write a bank; rows must be ordered by split, cohort question order, then sample index."""
    if len(table) != len(embeddings):
        raise ValueError("Embedding rows do not match responses")
    table = table.cast(BANK_SCHEMA).replace_schema_metadata({k: str(v) for k, v in metadata.items()})
    path = bank_path(model, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", row_group_size=4096)
    np.save(embedding_path(model, dataset), np.ascontiguousarray(embeddings, dtype=np.float32), allow_pickle=False)


def load_top_logprobs(model: str, dataset: str, split: str):
    """Per-token top-K log-probabilities of every response in a split (optional download: ``pgm download --only logprobs``).

    Returns a list in bank order (cohort question order, then sample index) of (token_ids, logprobs), each
    an array [n_tokens, K] (K = 20, or 21 when the sampled token was outside the top 20; padding is -1 / -inf).
    """
    path = bank_path(model, dataset).with_suffix(".top_logprobs.parquet")
    table = pq.read_table(path, filters=[("split", "=", split)])
    out = []
    for k, ids, lps in zip(table["top_k"].to_pylist(), table["top_token_ids"], table["top_logprobs"]):
        ids = ids.values.to_numpy()
        out.append((ids.reshape(-1, k) if k else ids.reshape(0, 0),
                    lps.values.to_numpy(zero_copy_only=False).reshape(-1, k) if k else np.zeros((0, 0), np.float32)))
    return out
