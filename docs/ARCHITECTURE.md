# Package architecture

`pgm_asc`는 구현을 역할별 subpackage로 나눈다. Root에는 `__init__.py`, `__main__.py`, `cli.py`만 둔다. 공개 실행 명령은 계속 `python -m pgm_asc`다.

| Folder | Responsibility | Main files |
|---|---|---|
| `core/` | 모든 단계가 공유하는 config, JSON I/O, answer equivalence, integrity, runtime | `config.py`, `io.py`, `answers.py`, `integrity.py`, `runtime.py` |
| `data/` | Response bank를 causal prefix 입력으로 구성하고 feature·target·scaler 제공 | `banks.py`, `state.py`, `features.py`, `prefixes.py`, `scaling.py` |
| `models/` | PyTorch model·partition 정의와 forward prediction, frozen Ridge value prediction | `selector.py`, `state.py`, `partitions.py`, `value.py` |
| `training/` | Supervised fitting, prefix objective, question-level OOF, continuation target | `selector.py`, `state.py`, `continuation.py`, `value.py`, `recipes.py` |
| `policies/` | Temperature/bucket calibration과 현재 prefix에 기반한 stopping decision | `calibration.py`, `stopping.py`, `continuation.py`, `baselines.py` |
| `collection/` | Cohort 선정, vLLM generation, teacher-forcing, confidence extraction, GPU sharding | `banks.py`, `vllm.py`, `teacher_forcing.py`, `confidence.py`, `token_stats.py` |
| `evaluation/` | Gold와 decision을 결합한 replay metric, locked holdout 평가, report | `selector.py`, `state.py`, `replay.py`, `metrics.py`, `report.py` |
| `experiments/` | 두 stage의 실행 순서, calibration policy 선택, cache context와 integrity barrier | `pipeline.py`, `selector.py`, `state.py`, `selector_context.py`, `state_context.py`, `validation.py` |

## Find the implementation

- PGM equation이나 neural structure 변경: `models/selector.py`, `models/state.py`.
- Partition grouping 변경: `models/partitions.py`.
- Input feature와 train-only scaling 변경: `data/`.
- Loss·optimizer·OOF 변경: `training/`.
- Stop/continue rule 또는 calibration 변경: `policies/`.
- Response sampling·hidden state extraction 변경: `collection/`.
- Stage별 실험 후보·순서·barrier 변경: `experiments/`.
- Holdout 결과·bootstrap·그림 변경: `evaluation/`.

`models/`, `data/`, `policies/`, `core/`는 training loop나 experiment runner를 import하지 않는다. Training/evaluation이 공통 model·feature·policy 함수를 사용한다. 같은 함수의 복사본이나 기존 flat module용 wrapper를 남기지 않는다.

`selector_context.py`와 `state_context.py`는 각 stage의 filesystem cache, frozen checkpoint, prediction을 묶는다. 실제 holdout 평가에서는 이미 선택된 policy를 replay한다. Calibration-A/B의 supervised fitting과 policy 선택은 `experiments/selector.py`, `experiments/state.py`에서 수행한다.

## Runtime and data

설정은 `configs/experiment.json` 하나다. `core/config.py`가 공통 설정과 stage 설정을 합치고 project root를 해석한다. `.venv`는 학습·분석용이며, collection runner는 vLLM만 `.venv-vllm` subprocess로 실행한다.

```bash
.venv/bin/python -m pgm_asc status
.venv/bin/python -m pgm_asc run
.venv/bin/python -m pgm_asc validate
.venv/bin/python -m pytest -q
```

하위 collection 모듈을 직접 디버깅할 때는 `python -m pgm_asc.collection.vllm`, `python -m pgm_asc.collection.teacher_forcing`, `python -m pgm_asc.collection.confidence`를 사용한다. 일반 실행에서는 root CLI의 `collect`가 config·GPU·subprocess를 연결한다.

데이터와 checkpoint는 기존 `artifacts/data/pgm_asc`, `artifacts/runs/pgm_asc`에 유지한다. 코드를 다시 정리해도 cohort ID, sample seed, weights, frozen policy를 변경하지 않는다. `excluded_problem_hashes.json`은 모든 과거 cohort의 normalized identity를 보존하는 공통 데이터이며 코드 폴더와 독립적이다.

## Integrity and tests

새 lock을 만들 때 `core.integrity.source_files()`로 모든 하위 `.py` 파일을 포함한다. 기존 lock을 구조 변경에 맞춰 이관할 때는 이전 lock을 먼저 검증하고 checkpoint·training input·cohort hash를 유지한다. 이관 전 protocol hash와 변경 이유는 기록으로 남긴다. 이관된 lock은 새로운 preregistration을 뜻하지 않는다.

`tests/unit/`는 model, policy, answer, feature의 수치·causal invariance를 확인한다. `tests/integration/`는 package 의존성, recursive hash coverage, restricted checkpoint loading, subprocess 경로를 확인한다. 실제 bank replay 검증은 `artifacts/runs/pgm_asc/package_layout_validation.json`에 기록한다.
