"""Resumable per-question workspace used while collecting a bank (artifacts/work/<model>_<dataset>/<split>/).

Stages communicate through small JSON files so generation, feature extraction and judging can run
concurrently on different GPUs and resume after interruption. ``pgm collect pack`` turns a
finished workspace into the two bank files.
"""
from __future__ import annotations

from pathlib import Path

from pgm_asc.config import ARTIFACTS, policy_path
from pgm_asc.utils import read_json, write_json


class Workspace:
    def __init__(self, model: str, dataset: str, split: str):
        self.root = ARTIFACTS / "work" / f"{model}_{dataset}" / split
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, kind: str, name: str) -> Path:
        return self.root / kind / f"{name}.json"

    def done(self, kind: str, name: str) -> bool:
        return self.path(kind, name).exists()

    def read(self, kind: str, name: str):
        return read_json(self.path(kind, name))

    def write(self, kind: str, name: str, value) -> None:
        write_json(self.path(kind, name), value)

    def names(self, kind: str) -> set[str]:
        folder = self.root / kind
        return {p.stem for p in folder.glob("*.json")} if folder.exists() else set()


def check_test_allowed(model: str, dataset: str, split: str, allow_test: bool) -> None:
    """Test banks are collected only after a policy has been fitted (no test data before selection)."""
    if split != "test" or allow_test:
        return
    candidates = [f"{model}_{dataset}"] + ([f"{model}_math"] if dataset == "omnimath" else [])
    if not any(policy_path(p).exists() for p in candidates):
        raise SystemExit(f"Fit a policy before collecting the {model}/{dataset} test bank (or pass --allow-test).")
