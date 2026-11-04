"""Download the released artifacts from the Hugging Face Hub and verify them against manifest.json."""
from __future__ import annotations

import os

from pgm_asc.config import ARTIFACTS, MODELS, STUDY
from pgm_asc.utils import file_sha256, read_json, utc_now, write_json

HUB_DATASET = os.environ.get("PGM_HUB_DATASET", STUDY["hub_dataset"])
FOLDERS = ("cohorts", "banks", "policies", "decisions")
# "logprobs" (per-token top-20 log-probabilities, 18 GB) is opt-in: nothing in the study reads it.
KINDS = FOLDERS + ("logprobs",)
DEFAULT_KINDS = FOLDERS


def patterns(kinds=DEFAULT_KINDS, models=None, datasets=None, embeddings=True):
    models = models or list(MODELS)
    datasets = datasets or list(STUDY["datasets"])
    out = ["manifest.json"]
    for kind in kinds:
        if kind == "cohorts":
            out += [f"cohorts/{d}.parquet" for d in datasets]
        elif kind == "banks":
            out += [f"banks/{m}_{d}.parquet" for m in models for d in datasets]
            if embeddings:
                out += [f"banks/{m}_{d}.embeddings.npy" for m in models for d in datasets]
        elif kind == "policies":
            out += [f"policies/{m}_{d}.*" for m in models for d in datasets]
        elif kind == "decisions":
            out += [f"decisions/{m}_*.parquet" for m in models]
        elif kind == "logprobs":
            out += [f"banks/{m}_{d}.top_logprobs.parquet" for m in models for d in datasets]
    return out


def download(kinds=DEFAULT_KINDS, models=None, datasets=None, embeddings=True, revision=None):
    from huggingface_hub import snapshot_download
    snapshot_download(HUB_DATASET, repo_type="dataset", revision=revision, local_dir=ARTIFACTS,
                      allow_patterns=patterns(kinds, models, datasets, embeddings))
    return verify(missing_ok=True)


def build_manifest():
    """Record size and SHA256 of every released file (run before uploading)."""
    files = {}
    for kind in FOLDERS:
        for path in sorted((ARTIFACTS / kind).glob("*")):
            if path.is_file():
                files[f"{kind}/{path.name}"] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    write_json(ARTIFACTS / "manifest.json", {"created_at": utc_now(), "files": files})
    return files


def verify(missing_ok=False):
    manifest = read_json(ARTIFACTS / "manifest.json")["files"]
    present, bad = 0, []
    for name, entry in manifest.items():
        path = ARTIFACTS / name
        if not path.exists():
            if not missing_ok:
                bad.append(f"missing {name}")
            continue
        present += 1
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            bad.append(f"changed {name}")
    if bad:
        raise SystemExit("Artifact verification failed:\n  " + "\n  ".join(bad))
    return present, len(manifest)
