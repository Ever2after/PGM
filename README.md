# PGM Adaptive Self-Consistency

Self-consistency samples a language model many times and takes a majority vote. Adaptive variants stop
once the vote looks settled. This repository studies a small **probabilistic graphical model (PGM) head**
that aggregates the responses seen so far into a distribution over answers (plus "not sampled yet") and
stops when it is confident. The head discounts redundant reasoning through a mixture over semantic
partitions of the responses.

The study compares the PGM mixture against **11 matched ablations** and the **SC, ASC, ESC and ReASC**
baselines. It covers **Gemma-3-4B-IT, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct**
on **MATH500, GSM8K and Omni-MATH**, plus MATH → Omni-MATH transfer. Language models stay frozen; every
method replays the same 16 responses per question.

[Method](docs/METHOD.md) · [All results](results/README.md) · [Interactive report](results/report.html) · [Data format](docs/DATA.md) · [Reproducing](docs/REPRODUCING.md) · [Artifacts on 🤗](https://huggingface.co/datasets/Ever2after/PGM-artifacts)

## Results

Accuracy / mean output tokens per question (seed 42). PGM mixture and ASC dev-tuned use thresholds
chosen on dev (fewest tokens with dev accuracy ≥ dev SC16 − 1 pp).

| Model | Evaluation | SC16 | ASC dev-tuned | ReASC offline | PGM mixture |
|---|---|---:|---:|---:|---:|
| Gemma-3-4B-IT | MATH500 | 77.4 / 11,852 | 78.0 / 4,161 | 76.8 / 5,648 | 77.8 / 4,571 |
| Gemma-3-4B-IT | GSM8K | 91.8 / 7,144 | 91.7 / 1,325 | 91.6 / 1,807 | 91.7 / 1,209 |
| Gemma-3-4B-IT | Omni-MATH | 31.6 / 21,542 | 31.9 / 11,047 | 31.7 / 14,346 | 32.2 / 19,020 |
| Gemma-3-4B-IT | MATH → Omni | 31.6 / 21,542 | 31.9 / 9,456 | 31.3 / 15,277 | 30.9 / 7,576 |
| Qwen2.5-3B-Instruct | MATH500 | 73.0 / 9,628 | 73.4 / 5,586 | 72.4 / 3,847 | 74.2 / 5,627 |
| Qwen2.5-3B-Instruct | GSM8K | 90.4 / 4,889 | 85.3 / 302 | 89.0 / 1,012 | 88.7 / 775 |
| Qwen2.5-3B-Instruct | Omni-MATH | 28.0 / 13,221 | 27.4 / 7,529 | 27.4 / 9,296 | 27.9 / 11,385 |
| Qwen2.5-3B-Instruct | MATH → Omni | 28.0 / 13,221 | 28.0 / 10,272 | 27.6 / 9,743 | 28.3 / 11,785 |
| Qwen2.5-7B-Instruct | MATH500 | 79.8 / 9,111 | 79.6 / 2,789 | 78.0 / 3,112 | 79.2 / 2,949 |
| Qwen2.5-7B-Instruct | GSM8K | 94.1 / 4,635 | 93.8 / 707 | 93.9 / 983 | 93.6 / 650 |
| Qwen2.5-7B-Instruct | Omni-MATH | 33.5 / 13,156 | 33.3 / 6,618 | 32.5 / 8,335 | 34.1 / 10,715 |
| Qwen2.5-7B-Instruct | MATH → Omni | 33.5 / 13,156 | 33.3 / 6,618 | 32.7 / 9,114 | 33.0 / 7,390 |
| Llama-3.1-8B-Instruct | MATH500 | 58.8 / 7,709 | 59.6 / 3,997 | 58.4 / 4,311 | 60.8 / 4,062 |
| Llama-3.1-8B-Instruct | GSM8K | 90.4 / 2,885 | 89.5 / 609 | 88.9 / 751 | 89.4 / 600 |
| Llama-3.1-8B-Instruct | Omni-MATH | 17.2 / 16,941 | 17.0 / 12,171 | 17.9 / 14,409 | 17.4 / 15,988 |
| Llama-3.1-8B-Instruct | MATH → Omni | 17.2 / 16,941 | 17.0 / 12,171 | 17.1 / 14,986 | 17.1 / 14,200 |

- On MATH and GSM8K the mixture stays within −1.7 to +2.0 pp of SC16 with 42–86 % fewer output tokens.
  Dev-tuned ASC reaches a similar point.
- On Omni-MATH the in-domain heads are conservative: −0.1 to +0.6 pp of SC16, with only 6–19 % fewer
  tokens. The MATH policies transferred to Omni-MATH stop earlier for three of four models.
- **The semantic partition structure is not supported.** One of 48 in-domain primary structure
  contrasts favors the mixture after per-cell Holm adjustment (Llama GSM8K vs. the confidence MLP, at
  12 % more tokens). Size-matched random grouping is never different from the mixture: the learned
  cluster-size discount (α ≈ 0.45) helps, but which responses are grouped together does not.
- Output tokens include both answer-extraction passes and exclude input processing and the
  teacher-forcing feature pass, so they are not serving-time speedups.

![PGM mixture vs. baselines, in-domain](results/figures/pgm_vs_baselines_deltas.png)

## Quick start

```bash
git clone https://github.com/Ever2after/PGM && cd PGM
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'      # CPU is enough for everything except collecting new responses
pgm download                  # cohorts, banks, policies, decisions (~7 GB) into artifacts/
pgm verify                    # SHA256 against manifest.json
pgm evaluate all              # replay all 16 cells (about 5 min) -> results/evaluations/
pgm report                    # results/README.md, summary.csv, figures/, report.html
```

| Goal | Command | Needs |
|---|---|---|
| Inspect results | open `results/` | nothing |
| Replay every reported number | `pgm download && pgm evaluate all` | CPU |
| Refit the heads of a cell (36 heads, ~10 min) | `pgm fit qwen7b_math --overwrite` | CPU |
| Try a new aggregation head | add an arm to `pgm_asc/model.py`, then `pgm fit` / `pgm evaluate` | CPU |
| Collect a bank for a new model | add it to `configs/models.json`, then `pgm collect run --model ... --dataset ...` | 4 GPUs, `pip install -e '.[collect]'` |

`pgm status` shows which banks, policies and evaluations are present. On a Slurm cluster use
`scripts/slurm/cpu.sbatch` and `scripts/slurm/collect.sbatch`. See [docs/REPRODUCING.md](docs/REPRODUCING.md).

## What is where

| | GitHub (this repository) | Hugging Face [`Ever2after/PGM-artifacts`](https://huggingface.co/datasets/Ever2after/PGM-artifacts) |
|---|---|---|
| Code | `pgm_asc/` package, CLI, tests, Slurm scripts | – |
| Configuration | `configs/models.json` (pinned models, sampling), `configs/study.json` (protocol) | – |
| Questions and gold answers | – | `cohorts/` (3 files) |
| Sampled responses, features, labels, embeddings | – | `banks/` (24 files) |
| Per-token top-20 log-probabilities (optional) | – | `banks/*.top_logprobs.parquet` (12 files, 18 GB) |
| Fitted heads, calibrators, thresholds | – | `policies/` (24 files) |
| Per-question decisions | – | `decisions/` (16 files) |
| Aggregate metrics, statistics, figures, report | `results/` | – |
| Documentation | `README.md`, `docs/` | dataset card |

```text
pgm_asc/
  config.py        models, study protocol, the 16 cells, artifact paths
  data/            cohorts and banks (bank.py), online partitions, prefix features
  model.py         PGM head and all ablation arms
  train.py         training loop (weighted prefix NLL, early stopping)
  policies.py      temperature calibration, stopping rule, dev threshold selection
  baselines.py     SC, ASC, ESC, ReASC offline/online, dev-tuned ASC/ReASC
  fit.py           fit every head of a cell -> policies/
  evaluate.py      replay policies and baselines -> results/evaluations/, decisions/
  stats.py         decision rows, metrics, paired bootstrap, Holm
  collect/         GPU pipeline: generation, teacher forcing, xVerify judge, packing
  report/          tables, figures, report.html
  hub.py, cli.py   download/verify, command line
configs/  results/  docs/  scripts/  tests/
```

## References

- Self-Consistency ([Wang et al., ICLR 2023](https://arxiv.org/abs/2203.11171)), Adaptive-Consistency
  ([Aggarwal et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.761/)), Early-Stopping
  Self-Consistency ([Li et al., ICLR 2024](https://arxiv.org/abs/2401.10480)), ReASC
  ([Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1085/); [code](https://github.com/junseokkim00/ReASC) @ `8ee1c76`).
- Judge: xVerify-3B-Ia ([paper](https://arxiv.org/abs/2504.10481), [code](https://github.com/IAAR-Shanghai/xVerify) @ `3f50a37`).
  The original ReASC judge alias was unavailable, so scores are a controlled reproduction, not identical to published numbers.
- Data: [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) and the
  [PRM800K MATH500 split](https://github.com/openai/prm800k#math-splits), [GSM8K](https://huggingface.co/datasets/openai/gsm8k),
  [Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH). Upstream data, models and code keep their own licenses.

The pre-release history, including the earlier V2/V5 selector/state/value experiments, is at git tag `v0-archive`.
