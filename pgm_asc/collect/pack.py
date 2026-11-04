"""Stage 4 (CPU): pack a finished workspace into the bank files (parquet + aligned embeddings)."""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

from pgm_asc.collect.workspace import Workspace
from pgm_asc.data.bank import BANK_SCHEMA, N_SAMPLES, SPLIT_ORDER, exact_groups, load_cohort, write_bank
from pgm_asc.utils import log


def flatten(r, split, correct, raw_correct):
    row = {k: r.get(k) for k in BANK_SCHEMA.names if k in r}
    row.update(split=split, sampled_logprobs=r["native_sample_logprobs"], self_certainty=r["self_certainty_per_token"],
               correct=correct, raw_correct=raw_correct)
    for phase in ("inline", "final"):
        for k in ("input_tokens", "output_tokens", "seconds_share"):
            row[f"extraction_{phase}_{k}"] = r[f"extraction_{phase}"][k]
    return row


def pack(model, dataset, splits=SPLIT_ORDER):
    rows, vectors, identities = [], [], {"judge_identity": set(), "generation_identity": set()}
    packed = []
    for split in splits:
        ws = Workspace(model, dataset, split)
        questions = load_cohort(dataset, split)
        missing = [q["question_id"] for q in questions if not ws.done("judgments", q["question_id"])]
        if missing:
            log("split not packed (incomplete)", model=model, dataset=dataset, split=split, missing=len(missing))
            continue
        for q in questions:
            qid = q["question_id"]
            judgment = ws.read("judgments", qid)
            records = [ws.read("scored", f"{qid}__s{i:03d}") for i in range(N_SAMPLES)]
            reps, groups = exact_groups([r["prediction"] for r in records])
            if reps != judgment["representatives"]:
                raise ValueError("Judged candidates differ from exact answer groups: " + qid)
            for r, g, raw in zip(records, groups, judgment["raw_response_correct"]):
                rows.append(flatten(r, split, bool(judgment["canonical_correct"][g]), bool(raw)))
                vectors.append(np.load(ws.root / "embeddings" / f"{r['response_id']}.npy"))
                identities["generation_identity"].add(r["generation_identity"])
            identities["judge_identity"].add(judgment["judge_identity"])
        packed.append(split)
    if not packed:
        raise SystemExit("Nothing to pack: no split has complete judgments.")
    table = pa.Table.from_pylist(rows, schema=BANK_SCHEMA)
    write_bank(model, dataset, table, np.stack(vectors),
               {"model": model, "dataset": dataset, "splits": json.dumps(packed), **{k: json.dumps(sorted(v)) for k, v in identities.items()}})
    log("bank packed", model=model, dataset=dataset, splits=packed, responses=len(rows))
