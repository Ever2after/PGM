"""Fix the in-domain Omni-MATH fitting cohorts and per-model cell configs.

Fitting questions come from the full Omni-MATH release, excluding every question
of the existing 999-question Omni test cohort (normalized problem identity). The
test cohort and its gold sidecar are copied byte-for-byte, so the in-domain cell
and the MATH-transfer cell are evaluated on the same questions and responses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pgm_asc.core.integrity import file_sha, problem_key  # noqa: E402
from pgm_asc.core.io import read_json, read_jsonl, utc_now, write_json, write_jsonl  # noqa: E402

SOURCE = "KbsdJames/Omni-MATH"
DATA = Path("artifacts/data/pgm_followup/omnimath_indomain")
TEST = Path("artifacts/data/pgm_followup/omnimath")
SEED = 20261103
SPLITS = (("calibration", 128), ("train", 800), ("dev", 200))
MODELS = ("gemma4b", "qwen3b", "qwen7b")


def prepare(source: Path, revision: str) -> dict:
    manifest = DATA / "manifest.json"
    if manifest.exists():
        recorded = read_json(manifest)
        for path, sha in recorded["artifact_sha256"].items():
            if file_sha(path) != sha:
                raise ValueError("Changed cohort " + path)
        return recorded
    test = read_jsonl(TEST / "test.jsonl")
    test_keys = {q["problem_key"] for q in test}
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    eligible, seen, overlaps, duplicates = [], set(), 0, 0
    for index, row in enumerate(rows):
        key = problem_key(row["problem"])
        if key in test_keys:
            overlaps += 1
        elif key in seen:
            duplicates += 1
        else:
            seen.add(key)
            eligible.append((index, key, row))
    order = np.random.default_rng(SEED).permutation(len(eligible))
    DATA.mkdir(parents=True, exist_ok=True)
    offset, counts = 0, {}
    for split, n in SPLITS:
        chosen = [eligible[i] for i in order[offset:offset + n]]
        offset += n
        write_jsonl(DATA / f"{split}.jsonl", [
            {"question_id": f"study_omnimath_{split}_{key[:16]}", "problem": row["problem"], "source_split": "test",
             "source_id": f"omni_math_row_{index}", "problem_key": key, "domain": row["domain"],
             "difficulty": row["difficulty"], "source": row["source"]} for index, key, row in chosen])
        write_jsonl(DATA / f"{split}_gold.jsonl", [
            {"question_id": f"study_omnimath_{split}_{key[:16]}", "gold_answer": row["answer"]} for index, key, row in chosen])
        counts[split] = len(chosen)
    for name in ("test.jsonl", "test_gold.jsonl"):
        shutil.copyfile(TEST / name, DATA / name)
    counts["test"] = len(test)
    files = sorted(DATA.glob("*.jsonl"))
    result = {"created_at": utc_now(), "source": SOURCE, "source_revision": revision, "source_rows": len(rows),
              "selection_seed": SEED, "counts": counts, "eligible_fitting_questions": len(eligible),
              "rows_overlapping_test_removed": overlaps, "duplicate_rows_removed": duplicates,
              "test_cohort_copied_from": str(TEST / "test.jsonl"),
              "note": "Omni-MATH has no official train split; fitting cohorts are disjoint from the 999-question test cohort.",
              "artifact_sha256": {str(p): file_sha(p) for p in files}}
    write_json(manifest, result)
    return result


def configs() -> None:
    for model in MODELS:
        cfg = read_json(f"configs/pgm_followup/{model}_omnimath.json")
        cfg.pop("transfer_from")
        cfg.update(run_dir=f"artifacts/runs/pgm_followup/{model}_omnimath_indomain", data_dir=str(DATA),
                   label="PGM necessity study, in-domain Omni-MATH fit: matched supervision; no continuation; pinned official-model defaults")
        path = Path(f"configs/pgm_followup/{model}_omnimath_indomain.json")
        if not path.exists():
            write_json(path, cfg)


def link_test_bank() -> None:
    """Share the transfer cell's test responses read-only; evaluation outputs stay separate."""
    for model in MODELS:
        target = Path(f"artifacts/runs/pgm_followup/{model}_omnimath_indomain/test")
        target.mkdir(parents=True, exist_ok=True)
        for name in ("questions", "responses", "scored_responses", "embeddings", "judgments", "top_logprobs"):
            link = target / name
            if not link.exists():
                # Relative, so the artifact tree can move with the repository.
                link.symlink_to(Path("../..") / f"{model}_omnimath" / "test" / name, target_is_directory=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Omni-MATH test.jsonl from the pinned HF revision")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    import os
    os.chdir(PROJECT)
    print(json.dumps({k: v for k, v in prepare(args.source, args.revision).items() if k != "artifact_sha256"}, indent=1))
    configs()
    link_test_bank()
