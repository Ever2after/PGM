# Reproducing and extending the study

## Environment

Python 3.10. `pip install -e '.[test]'` is enough for downloading, replaying, refitting and reporting
(CPU only). Collecting new responses needs `pip install -e '.[collect]'` (vLLM 0.11.0, Transformers 4.56.2),
CUDA GPUs and the model weights; the exact recorded environment is `requirements.txt`. On aarch64
(e.g. GH200) PyTorch 2.8.0 exists only as the CUDA 12.9 build:
`pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129`.

Set `PGM_ARTIFACTS` to keep the artifacts outside the repository (default `./artifacts`).

## 1. Replay the reported results (CPU, minutes)

```bash
pgm download            # or: pgm download --only cohorts banks policies --models qwen7b
                        # per-token top-20 log-probabilities are separate: pgm download --only logprobs
pgm verify
pgm evaluate all        # or one cell: pgm evaluate llama8b_math_to_omnimath
pgm report
```

`evaluate` loads a saved policy, rebuilds the prefix features from the bank, replays all 36 heads at both
operating points and all 8 baselines, and writes `results/evaluations/<cell>.json` and
`artifacts/decisions/<cell>.parquet`.

## 2. Refit policies (CPU, ~10 min per cell)

```bash
pgm fit qwen7b_math --overwrite                # 11 arms x 3 seeds + same-head control
pgm evaluate qwen7b_math && pgm evaluate qwen7b_math_to_omnimath
```

Transfer cells have no policy of their own; they replay `<model>_math`. Training uses 4 CPU threads and
fixed seeds, so it is deterministic on a given machine (see [Verification](#verification)).

## 3. Collect a new bank (GPU)

```bash
python scripts/setup_references.py            # pinned xVerify prompts for the judge
pgm collect run --model qwen7b --dataset math  # one node, GPUs 0-3
```

`run` collects calibration/train/dev (two vLLM generation shards, a teacher-forcing worker and the
xVerify judge run concurrently on four GPUs), packs them into the bank, fits the policy, then collects and
packs the test split and evaluates. The test bank is only collected after a policy exists. Every stage is
resumable: rerun the same command after an interruption. Stages can also be run separately with
`pgm collect generate|features|judge|pack --split ...`. Work files go to `artifacts/work/`.

On Slurm: `sbatch -J qwen7b-math scripts/slurm/collect.sbatch --model qwen7b --dataset math` and
`sbatch scripts/slurm/cpu.sbatch fit qwen7b_math`. The scripts are written for KISTI Hangang (4×GH200
nodes); adjust partition, comment and time limits for other clusters.

Runtime notes from the recorded runs:

- Each model's own generation defaults are used and checked against `configs/models.json`.
- Llama-3.1-8B on Omni-MATH needs `--memory 0.5`: with 0.25 of a GPU the KV cache (~63k tokens) causes
  preemption, which trips a vLLM 0.11 prompt-logprob assertion. Sampling and seeds are unaffected.
- Tokenizers without a pad token (Llama 3.1) use EOS for padding in the teacher-forcing pass; padding is
  attention-masked, so features are unchanged.
- The judge context is 16,384 tokens to fit long Omni-MATH responses.

**Adding a model**: add an entry to `configs/models.json` (Hub id, revision, its default sampling
parameters, rounded parameter count). **Adding a dataset**: write `artifacts/cohorts/<name>.parquet` with
`split ∈ {calibration, train, dev, test}`, `question_id`, `problem`, `gold_answer`, and add it to
`datasets` and `tasks` in `configs/study.json`. **Adding a head**: add an arm to `ARMS` in
`pgm_asc/model.py` (or a new `mode` in `PGM.forward`); `pgm fit` trains it like every other arm.

## Verification

The released files were repacked from the original run directories (about one million JSON/NPZ files)
and the code was rewritten. Checks on 2026-11-05 (KISTI Hangang, aarch64):

| Check | Result |
|---|---|
| Bank packing | Every split: responses = embeddings = 16 × judged questions; exact-answer groups equal the judged candidates for every question |
| Replay, 10 cells whose originals were computed on this machine (all Omni-MATH and Llama cells) | Evaluation JSON and per-question decisions identical to the original, bit for bit |
| Replay, 6 cells first computed on the original x86 machine (Gemma/Qwen MATH and GSM8K) | Every per-question decision of every reported method identical; differences only in floating-point rounding (relative ≤ 1.4e-7 in NLLs, last-digit differences in exact binomial CIs and ReASC-online GMM parameters) and one question in one unreported threshold curve (qwen7b MATH, DeepSets-confidence seed 44, γ = 0.995) |
| Refit `llama8b_math` (trained on this machine) | All 36 heads bit-identical; scalers, temperatures, selected thresholds and baseline calibration identical |
| Refit `gemma4b_math`, `qwen7b_math` (trained on x86) | Selected thresholds with their dev accuracy and tokens, and best epochs, identical for all 36 heads; weights differ at floating-point level because feature scalers differ by ~1e-5 relative across CPU architectures |

So the released policies reproduce every reported number, and refitting reproduces the same operating
points; bitwise-identical weights require the same CPU architecture.

## Known limitations

- Output-token cost excludes input processing and the teacher-forcing pass (reported separately in the
  evaluation JSON as `feature_input_tokens`, `extraction_input_tokens`, `dense_2N_approx_T`). Replay time
  is not serving latency.
- xVerify-3B-Ia replaces the unavailable original ReASC judge; it mislabels some answers (e.g. a sign error
  judged correct). All methods share the same labels.
- ReASC online calibration fits its GMM on the test questions' first responses (label-free but
  transductive), as in the reference implementation.
- Holm adjustment covers the four primary contrasts within one cell, not the whole matrix.
