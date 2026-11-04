# Reproduction guide

## What is included

Git contains the implementation, tests, recorded configs/protocols, aggregate
results for all completed evaluations, and cohort/source/selection hashes.
`results/followup/metrics.json` includes all training seeds, both operating
points, fixed-prefix metrics, and paired contrasts.

Git excludes raw benchmark questions, generated responses, per-question judge
outputs, embeddings, checkpoints, downloaded models, local environments and
logs. Existing local files under `artifacts/` remain available on the experiment
machine. This repository is a code/aggregate-results release; it is not a full
historical artifact bundle.

The original experiment locks reference those local paths. Copying a lock to
`results/followup/provenance/` preserves evidence; it does not make missing model
weights or banks available.

## Environments

| Purpose | Environment / requirements |
|---|---|
| Model development and synthetic tests | `pip install -e '.[test]'`; no vLLM required |
| ReASC comparison and follow-up | Python 3.10.12; `requirements-reasc.txt` |
| Historical integrated V2/V5 analysis | `requirements.txt`, host PyTorch/CUDA |
| Historical V2/V5 generation | Separate environment using `requirements-vllm.txt` |

The latter two files are retained unchanged because they are included in
historical hashes. They are not the dependencies of the current vLLM study.
Editable installation is intended for use from the repository root; experiment
paths and frozen configs are repository-relative.

## Upstream references

```bash
python scripts/setup_references.py
python scripts/setup_references.py --check
```

This creates local ReASC/xVerify checkouts at exact commits, preserving upstream
licenses. It leaves existing checkouts untouched and fails on a different
revision or modified tracked files. See [third_party](../third_party/README.md).

Model IDs and revisions are in the per-cell configs. Download the pinned models
into `artifacts/hf_cache` before generation, including the generator and the
common xVerify judge. Accept any model-provider access terms and authenticate
locally where required; no token belongs in the repository. GPU workers use
offline cache access once assets are available.

## Restoring the recorded experiment

Exact historical replay requires the original artifact directories at their
recorded relative paths:

```text
artifacts/data/pgm_asc/excluded_problem_hashes.json
artifacts/data/reasc_pgm/                 # Original cohorts and separate gold
artifacts/data/pgm_followup/              # Follow-up cohorts and separate gold
artifacts/runs/pgm_asc/                   # Earlier frozen inputs, when used
artifacts/runs/reasc_pgm/                 # Initial banks / frozen preprocessing
artifacts/runs/pgm_followup/<cell>/        # Banks, heads, preprocessing, policy lock
artifacts/hf_cache/                      # Pinned model/dataset assets for GPU work
```

The initial PGM comparison inherits preprocessing from earlier frozen
experiments. Its original exclusion registry also affects cohort sampling.
Downloading MATH/GSM8K alone does not reconstruct all those historical inputs.
Do not run `pgm_asc.data.study` to overwrite the recorded configs or describe a
newly sampled cohort as the original holdout. A new experiment needs its own
artifact namespace and protocol.

Once original artifacts are restored:

```bash
python scripts/followup.py status

# Verify every registered input/checkpoint/source hash for one cell.
python - <<'PY'
from pgm_asc.core.integrity import verify_lock
from pgm_asc.core.io import read_json
verify_lock(read_json('artifacts/runs/pgm_followup/qwen7b_math/selection_lock.json'))
PY

# Fit from complete fitting banks, or validate/reuse an existing lock.
OMP_NUM_THREADS=4 python scripts/followup.py fit --cell qwen7b_math

# Replay the frozen policies; test data never selects a threshold.
OMP_NUM_THREADS=4 python scripts/followup.py evaluate --cell qwen7b_math --split confirm

# Refresh tracked aggregate results after completed evaluations.
python scripts/followup.py export
```

The exporter keeps evaluations already present in `results/followup/metrics.json`
when their source JSON is absent locally, so a partial artifact bundle can add
cells without dropping earlier ones. Point it at another bundle root with
`python scripts/export_results.py --project <bundle root>`.

The `fit` command trains small CPU heads, not the language model. `evaluate`
replays banks and checks the existing lock. Neither command generates missing
responses.

## GPU collection

For a prepared cohort/config, the generic collector has three explicit stages.
The following example uses physical GPUs 4–7, the allocation in the recorded
study. Run the four commands in separate terminals/processes:

```bash
CUDA_VISIBLE_DEVICES=4 python -m pgm_asc.collection.study collect \
  --config configs/pgm_followup/qwen7b_math.json --split confirm --shard 0 --shards 2
CUDA_VISIBLE_DEVICES=5 python -m pgm_asc.collection.study collect \
  --config configs/pgm_followup/qwen7b_math.json --split confirm --shard 1 --shards 2
CUDA_VISIBLE_DEVICES=6 python -m pgm_asc.collection.study features \
  --config configs/pgm_followup/qwen7b_math.json --split confirm --follow
CUDA_VISIBLE_DEVICES=7 python -m pgm_asc.collection.study judge \
  --config configs/pgm_followup/qwen7b_math.json --split confirm --follow
```

`scripts/run_cell.py --config <cell config>` runs these four workers itself
(GPU order: two generation shards, TF features, judge; `--gpus` remaps), for
calibration/train/dev, then fits and locks, then collects and replays each
evaluation split. Transfer cells skip collection. `--limit N` is a smoke test
that collects only the first N calibration questions. `scripts/slurm/cell.sbatch`
submits it as a whole-node job from the repository root.

Collection for evaluation splits requires the policy lock first. Use the same
commands with train/dev/calibration to prepare fitting banks before fitting.
Do not change shard count mid-run without reconciling generation manifests.
Completed shared Gemma banks are read-only links and must not be overwritten.

`experiments/study_runner*`, `qwen7_only.py`, and `math_priority.py` are retained
for the original supervised runs and their runtime amendments. Some reproduce
an earlier queue or require already fitted cells. The full original queue also
contains the Omni jobs that were deferred at the time; it is not the public default. Use the explicit
per-cell commands above for controlled work.

## Omni-MATH and Llama-3.1-8B cells (2026-11-03/04)

These cells were run on KISTI Hangang (one 4x GH200 aarch64 node per job),
starting from the private artifact bundle `Ever2after/PGM-omni-resume`
(Hugging Face dataset; snapshot of commit `9ec705a`) that holds the Gemma/Qwen
locked MATH cells and the fixed 999-question Omni cohort. Gemma/Qwen MATH and
GSM8K banks are not part of that bundle; their aggregates come from the original
machine and are retained in `results/followup/metrics.json`. The bundle's own
runners and manifest are archived in `scripts/omni_bundle/` and
`results/followup/provenance/omni_bundle_manifest.json`.

- **Omni transfer** (`<model>_omnimath`): each model's locked MATH cell is copied
  unchanged; no Omni label is used for fitting or selection.
- **Omni in-domain** (`<model>_omnimath_indomain`): Omni-MATH has no train split.
  `scripts/prepare_omni_indomain.py` draws calibration/train/dev (128/800/200)
  with seed 20261103 from the full release (`KbsdJames/Omni-MATH`, revision
  `40ba231`; 4,428 rows) after removing every test question by normalized problem
  identity and 14 duplicates. Its manifest is in
  `results/followup/provenance/omnimath_indomain_cohorts.json`. The test cohort
  file is copied byte-for-byte, and the in-domain and transfer cells share one test
  bank through read-only links, so they replay identical responses.
- **Llama-3.1-8B-Instruct** (`llama8b_*`, revision `0e9e39f`): native sampling
  from its generation config (temperature 0.6, top-p 0.9; top-k and repetition
  penalty absent, so vLLM 0.11 defaults 0 and 1.0), recorded in
  `provenance/llama8b_source_pin.json`. Cohorts are the same as for the other
  models; MATH is evaluated on MATH500 only.

Two runtime wrappers live in `scripts/` rather than `pgm_asc/`, so source hashes
registered in existing locks stay valid. `tf_worker.py` fills a missing pad token
with EOS: padding is right-sided and attention-masked, so features are unchanged,
and it is a no-op for Gemma/Qwen. `gen_worker.py` applies
`PGM_GPU_MEMORY_UTILIZATION`. At the code's fixed 0.25 fraction, Llama's Omni
generation got a 63k-token KV cache; preemption then tripped a vLLM 0.11
prompt-logprob assertion. That one cell ran with 0.5 after 216 train questions
had completed at 0.25. Sampling, seeds and request identities are unchanged.

PyTorch publishes 2.8.0 for aarch64 only as a CUDA 12.9 build, so
`torch==2.8.0+cu129` replaced the recorded CUDA 12.8 build; vLLM 0.11.0 and
Transformers 4.56.2 match the record. The test-time `math-verify` dependency comes
from the `[test]` extra, not `requirements-reasc.txt`.

## Interpretation

- In-domain and cross-dataset results are reported separately. Omni transfer
  operating points were selected on MATH dev only; in-domain Omni points on Omni dev.
- MATH500 matches the official ReASC benchmark problem set. Follow-up confirm500
  is a distinct subset of original MATH test, excluding historical cohorts.
- MATH train/dev are original-train questions. The inherited ReASC calibration
  sample is mixed: 79 original-train and 49 original-test questions, disjoint from
  evaluation. GSM8K fitting/calibration use official train only.
- All methods within a cell share samples, ordering, extraction and judge labels.
- Dev-selected policy minimizes output tokens subject to dev SC16 accuracy minus
  1 percentage point. This constraint is not a guarantee on test accuracy.
- Candidate-level xVerify-Ia accuracy is primary. This judge differs from the
  unavailable historical ReASC alias; scores are a controlled reproduction.
- Output-token savings omit input and extra TF costs. Offline replay time is not
  causal serving latency. Count-only controls do not require TF at serving.
- Holm adjustment covers four primary structure contrasts per model/dataset,
  not the whole experimental matrix. Training seeds are not extra test questions.

## GitHub publication

The configured repository name is `ever2after/PGM`. Create that empty repository
in GitHub, review the local changes, then commit and push:

```bash
git status --short
git add .
git diff --cached --stat
git commit -m "Prepare PGM adaptive self-consistency research repository"
git push -u origin main
```

The local preparation does not create a remote repository or publish files.
Virtual environments, model/data caches, credentials and downloaded upstream
repositories are excluded by `.gitignore`.
