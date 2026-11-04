"""A tiny synthetic study (cohorts + banks for one model/dataset) written to a temporary artifacts directory."""
import importlib
import json

import numpy as np
import pyarrow as pa
import pytest

N_PER_SPLIT = {"calibration": 6, "train": 10, "dev": 6, "test": 8}


def synthetic_records(rng, qid, split, n=16, dim=8):
    truth = str(int(rng.integers(0, 5)))
    rows, vectors = [], []
    for i in range(n):
        right = rng.random() < 0.6
        answer = truth if right else str(int(rng.integers(5, 8)))
        length = int(rng.integers(50, 400))
        rows.append({
            "split": split, "question_id": qid, "sample_index": i, "text": f"... \\boxed{{{answer}}}",
            "token_ids": list(range(5)), "sampled_logprobs": [-0.1] * 5, "self_certainty": [10.] * 5,
            "finish_reason": "stop", "truncated": False, "generated_token_count": length, "reference_chat_tokens": length + 40,
            "request_seed": i, "generation_identity": "synthetic", "native_generation_seconds_share": 0.1,
            "lowest_confidence": float(rng.normal(10 + right, 1)), "bottom10_confidence": float(rng.normal(11 + right, 1)),
            "extraction_inline_input_tokens": length + 60, "extraction_inline_output_tokens": 6, "extraction_inline_seconds_share": 0.01,
            "extraction_final_input_tokens": length + 60, "extraction_final_output_tokens": 6, "extraction_final_seconds_share": 0.01,
            "pred": f"\\boxed{{{answer}}}", "prediction": answer, "extraction_failed": False, "embedding_valid": True,
            "reasoning_mean_logprob": float(rng.normal(-0.3 + 0.1 * right, 0.05)), "answer_mean_logprob": float(rng.normal(-0.1, 0.05)),
            "reasoning_mean_entropy": float(rng.normal(0.4, 0.05)), "reasoning_token_count": length - 5, "answer_token_count": 1,
            "feature_input_tokens": length + 40, "feature_seconds_share": 0.01, "correct": right, "raw_correct": right})
        base = np.eye(dim)[int(answer) % dim]
        vectors.append(base + 0.3 * rng.normal(size=dim))
    return rows, np.asarray(vectors, np.float32)


@pytest.fixture(scope="session")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    import os
    # Never write into the repository: artifacts and results both go to temporary directories.
    os.environ["PGM_ARTIFACTS"] = str(root)
    os.environ["PGM_RESULTS"] = str(tmp_path_factory.mktemp("results"))
    import pgm_asc.config as config
    importlib.reload(config)
    config.STUDY["train"].update(epochs=2, patience=1)
    config.STUDY["train_seeds"] = [42, 43]
    for module in ("pgm_asc.data.bank", "pgm_asc.fit", "pgm_asc.evaluate"):
        importlib.reload(importlib.import_module(module))
    from pgm_asc.data.bank import write_bank
    rng = np.random.default_rng(0)
    cohort, rows, vectors = [], [], []
    for split, n in N_PER_SPLIT.items():
        for q in range(n):
            qid = f"q_{split}_{q}"
            r, v = synthetic_records(rng, qid, split)
            cohort.append({"split": split, "question_id": qid, "problem": "What is x?", "gold_answer": r[0]["prediction"]})
            rows += r
            vectors.append(v)
    path = config.cohort_path("math")
    path.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(cohort), path)
    write_bank("qwen3b", "math", pa.Table.from_pylist(rows), np.concatenate(vectors), {"model": "qwen3b", "dataset": "math", "splits": json.dumps(list(N_PER_SPLIT))})
    return config
