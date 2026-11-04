"""Study configuration, the 16 reported cells, and the artifact layout.

A *cell* is one model evaluated on one task. Tasks are in-domain (fit and evaluate on the
same dataset) or ``math_to_omnimath`` (the locked MATH policy replayed on Omni-MATH).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pgm_asc.utils import read_json

ROOT = Path(__file__).resolve().parents[1]
STUDY = read_json(ROOT / "configs" / "study.json")
MODELS = read_json(ROOT / "configs" / "models.json")
ARTIFACTS = Path(os.environ.get("PGM_ARTIFACTS", ROOT / "artifacts"))
RESULTS = Path(os.environ.get("PGM_RESULTS", ROOT / "results"))
FIT_SPLITS = tuple(STUDY["fit_splits"])
EVAL_SPLIT = "test"


@dataclass(frozen=True)
class Cell:
    model: str
    task: str

    @property
    def name(self) -> str:
        return f"{self.model}_{self.task}"

    @property
    def fit_dataset(self) -> str:
        return STUDY["tasks"][self.task]["fit"]

    @property
    def eval_dataset(self) -> str:
        return STUDY["tasks"][self.task]["evaluate"]

    @property
    def transfer(self) -> bool:
        return self.fit_dataset != self.eval_dataset

    @property
    def spec(self) -> dict:
        return MODELS[self.model]

    @property
    def policy_name(self) -> str:
        """The fitted policy this cell replays (its own, or the MATH policy for transfer)."""
        return f"{self.model}_{self.fit_dataset}"


def cells(model: str | None = None, task: str | None = None) -> list[Cell]:
    return [Cell(m, t) for m in MODELS for t in STUDY["tasks"]
            if (model is None or m == model) and (task is None or t == task)]


def fitted_cells() -> list[Cell]:
    return [c for c in cells() if not c.transfer]


def parse_cell(name: str) -> Cell:
    for cell in cells():
        if cell.name == name:
            return cell
    raise SystemExit(f"Unknown cell {name!r}; choose from: {', '.join(c.name for c in cells())}")


# Artifact layout (identical locally and in the Hugging Face dataset).
def cohort_path(dataset: str) -> Path:
    return ARTIFACTS / "cohorts" / f"{dataset}.parquet"


def bank_path(model: str, dataset: str) -> Path:
    return ARTIFACTS / "banks" / f"{model}_{dataset}.parquet"


def embedding_path(model: str, dataset: str) -> Path:
    return ARTIFACTS / "banks" / f"{model}_{dataset}.embeddings.npy"


def policy_path(policy_name: str) -> Path:
    return ARTIFACTS / "policies" / f"{policy_name}.json"


def checkpoint_path(policy_name: str) -> Path:
    return ARTIFACTS / "policies" / f"{policy_name}.checkpoints.pt"


def decisions_path(cell_name: str) -> Path:
    return ARTIFACTS / "decisions" / f"{cell_name}.parquet"


def evaluation_path(cell_name: str) -> Path:
    return RESULTS / "evaluations" / f"{cell_name}.json"
