# PGM Adaptive Self-Consistency

Research code for **probabilistic graphical models (PGM) for adaptive self-consistency**: aggregate a language model's reasoning samples and decide when to stop generating.

The current study compares a semantic partition mixture with matched structural ablations and ASC, ESC, and ReASC baselines across **Gemma-3-4B-IT, Qwen2.5-3B-Instruct, and Qwen2.5-7B-Instruct**, on MATH and GSM8K. Language models stay frozen; only small aggregation heads are trained.

[Results](results/followup/README.md) · [Experiment protocol](docs/PGM_FOLLOWUP_PLAN.md) · [Reproduction guide](docs/REPRODUCIBILITY.md) · [Documentation index](docs/README.md)

## Method

1. Generate a shared bank of up to **16 responses per question** with vLLM.
2. Extract exact-token confidence features and reasoning embeddings in a separate teacher-forcing pass.
3. Form several partitions of the observed responses, including cosine-similarity groups, all-one, and singletons.
4. Learn positive response support, a cluster-size discount, a partition mixture gate, and an **OTHER** probability for unseen answers.
5. Calibrate confidence on a separate cohort and stop when the selected policy is satisfied, with **minimum 2 / maximum 16** responses.

The follow-up uses **no continuation/value head**. Controls include uniform mixture, matched random partitions, alpha=1, independently trained single partition, count/confidence MLPs, and DeepSets. All learned arms share 800 training, 200 dev, and 128 calibration questions, with training seeds 42/43/44.

## Results

Primary seed 42; accuracy in %. Tokens are mean consumed output tokens, **including both answer-extraction outputs**, for the dev-selected mixture policy.

| Model | Evaluation | SC16 | ReASC offline | PGM mixture | Mixture output tokens |
|---|---|---:|---:|---:|---:|
| Gemma-3-4B-IT | MATH500 | 77.40 | 76.80 | 77.80 | 4,571 |
| Gemma-3-4B-IT | MATH confirm500 | 75.60 | 75.60 | 77.00 | 4,392 |
| Gemma-3-4B-IT | GSM8K | 91.81 | 91.58 | 91.74 | 1,209 |
| Qwen2.5-3B-Instruct | MATH500 | 73.00 | 72.40 | 74.20 | 5,627 |
| Qwen2.5-3B-Instruct | MATH confirm500 | 72.60 | 69.80 | 73.80 | 5,862 |
| Qwen2.5-3B-Instruct | GSM8K | 90.45 | 89.01 | 88.70 | 775 |
| Qwen2.5-7B-Instruct | MATH500 | 79.80 | 78.00 | 79.20 | 2,949 |
| Qwen2.5-7B-Instruct | MATH confirm500 | 79.60 | 77.80 | 78.80 | 3,059 |
| Qwen2.5-7B-Instruct | GSM8K | 94.09 | 93.86 | 93.56 | 650 |

The accuracy/cost trade-off varies across models and datasets. Current primary ablations **do not establish a significant accuracy advantage for semantic partition mixture** over the prespecified structure controls. Output-token savings exclude input processing and the extra feature-extraction pass; they are not measured serving speedups.

See [all methods, fixed95 operating points, and paired statistics](results/followup/README.md), or [machine-readable metrics](results/followup/metrics.json). MATH confirm500 is separate from standard MATH500 and all earlier registered cohorts. GSM8K uses all 1,319 official test questions. **Omni-MATH is deferred.**

## Setup and tests

For model/policy development and CPU tests (Python 3.10+):

```bash
git clone https://github.com/ever2after/PGM.git
cd PGM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python scripts/setup_references.py
OMP_NUM_THREADS=2 python -m pytest -q
```

`setup_references.py` downloads ReASC and xVerify at pinned commits. Tests use small synthetic banks and reference source files; they do not download model weights or run benchmark generation.

For the recorded vLLM study environment, use a **separate Python 3.10 environment**:

```bash
python3.10 -m venv .venv-reasc
source .venv-reasc/bin/activate
python -m pip install -r requirements-reasc.txt
python -m pip install --no-deps -e .
```

The study used vLLM 0.11.0, PyTorch 2.8.0, and Transformers 4.56.2. GPU generation requires a compatible NVIDIA/CUDA setup and the pinned model assets. Legacy `requirements.txt` and `requirements-vllm.txt` describe older V2/V5 environments; do not combine their pins with `requirements-reasc.txt`.

## Commands

Run commands from the repository root:

```bash
# Inspect local experiment completion; works without installing dependencies.
python scripts/followup.py status

# Inspect public aggregate results without downloading model weights.
python -m json.tool results/followup/metrics.json > /dev/null

# With the original local artifact bundle restored:
python scripts/followup.py fit --cell qwen7b_math
python scripts/followup.py evaluate --cell qwen7b_math --split confirm
python scripts/followup.py export
```

Fitting/replay require the corresponding banks, cohorts, preprocessing, and checkpoints under `artifacts/`. Those large artifacts are **not included in Git**. A fresh clone supports code inspection, tests, and aggregate analysis; exact historical training/replay additionally requires the original artifact bundle. The [reproduction guide](docs/REPRODUCIBILITY.md) describes these boundaries, generation commands, and integrity checks.

## Repository layout

```text
pgm_asc/
  core/           Configuration, I/O, answer handling, integrity
  data/           Cohorts, causal prefix features, scaling
  models/         PGM selector, matched ablation heads, legacy state/value
  training/       Training objectives and legacy training recipes
  policies/       Calibration, stopping, reference baselines
  collection/     vLLM generation, confidence/embedding extraction, judging
  evaluation/     Replay, cost metrics, paired statistics
  experiments/    Fit/lock/evaluate orchestration and historical runners
configs/          Recorded experiment settings and per-model/dataset cells
scripts/          Public follow-up commands, exports, reference setup
tests/           Unit tests and synthetic integration tests
results/followup/ Aggregate results and provenance tracked in Git
docs/             Protocols, implementation notes, experiment history
third_party/      Source audit metadata; downloaded checkouts are ignored
artifacts/        Local datasets, banks, model weights, logs (ignored)
```

The `python -m pgm_asc` CLI operates the **legacy integrated V2/V5 pipeline**. The current ReASC/follow-up study uses `scripts/followup.py` and the per-cell configs. Historical code and locked configs remain in their original paths so that their recorded hashes can still be verified.

## References and provenance

- [Official ReASC code](https://github.com/junseokkim00/ReASC); source differences and reproduction adaptations are documented in the [reference audit](docs/REASC_REFERENCE_AUDIT.md).
- [xVerify](https://github.com/IAAR-Shanghai/xVerify); a pinned public **xVerify-3B-Ia** judge is shared by every method because the original ReASC model alias was unavailable.
- [OpenAI PRM800K MATH splits](https://github.com/openai/prm800k#math-splits), [original MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math), and [GSM8K](https://huggingface.co/datasets/openai/gsm8k).

Model revisions, sampling settings, cohort hashes, and selection locks are retained in `configs/` and [results provenance](results/followup/provenance/). Upstream code, model weights, and datasets retain their respective licenses and access requirements.
