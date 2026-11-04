# Released data and artifacts

Everything needed to replay, refit or extend the study is in the Hugging Face dataset
[`Ever2after/PGM-artifacts`](https://huggingface.co/datasets/Ever2after/PGM-artifacts). The layout on the Hub and
under the local `artifacts/` directory is identical, so `pgm download` simply mirrors it. There are 79
files instead of one file per response.

```text
manifest.json                         size and SHA256 of every file (pgm verify)
cohorts/{math,gsm8k,omnimath}.parquet questions and gold answers, all splits
banks/<model>_<dataset>.parquet       16 responses per question, features, costs, labels
banks/<model>_<dataset>.embeddings.npy  float32 [n_responses, hidden], row-aligned with the parquet
banks/<model>_<dataset>.top_logprobs.parquet  per-token top-20 log-probabilities (optional download)
policies/<model>_<dataset>.json       fitted policy: representation, scalers, calibrators, thresholds, baselines
policies/<model>_<dataset>.checkpoints.pt  state dicts of the 36 heads
decisions/<cell>.parquet              per-question decision of every method in an evaluation
```

`<model>` ∈ `gemma4b qwen3b qwen7b llama8b`, `<dataset>` ∈ `math gsm8k omnimath`,
`<cell>` = `<model>_<task>` with `<task>` ∈ `math gsm8k omnimath math_to_omnimath`.

## Cohorts

One row per question: `split` (`calibration`, `train`, `dev`, `test`), `question_id`, `problem`,
`gold_answer`, and source fields (`source_split`, `source_id`, `problem_key` = SHA256 of the
NFKC-normalized problem; Omni-MATH also `domain`, `difficulty`, `source`). Row order is the order in
which questions are processed.

## Banks

Rows are ordered by split (calibration, train, dev, test), cohort question order, then `sample_index`
0–15. Row *i* of the embedding matrix belongs to row *i* of the parquet file.

| Columns | Meaning |
|---|---|
| `split`, `question_id`, `sample_index` | Identity |
| `text`, `token_ids`, `sampled_logprobs`, `finish_reason`, `truncated`, `generated_token_count` | Sampled response; `sampled_logprobs` are vLLM log-probabilities of the sampled tokens |
| `self_certainty` | Per-token ReASC self-certainty $-\log K - \frac1K\sum_k \log p_k$ over the $K$ log-probabilities returned by vLLM (top 20, plus the sampled token when outside them) |
| `lowest_confidence`, `bottom10_confidence` | Lowest / bottom-10 % mean over stride-1 windows of 128 tokens of `self_certainty` |
| `extraction_{inline,final}_{input,output}_tokens`, `..._seconds_share` | The two answer-extraction prompts |
| `pred`, `prediction` | Final extraction output and the parsed answer used for grouping |
| `reasoning_mean_logprob`, `answer_mean_logprob`, `reasoning_mean_entropy`, `reasoning_token_count`, `answer_token_count`, `extraction_failed`, `embedding_valid` | Teacher-forcing features (raw distribution, before temperature/top-p) |
| `feature_input_tokens`, `feature_seconds_share` | Cost of the teacher-forcing pass |
| `reference_chat_tokens`, `request_seed`, `generation_identity`, `native_generation_seconds_share` | Bookkeeping |
| `correct` | xVerify label of the response's exact-answer candidate (primary metric) |
| `raw_correct` | xVerify label of the raw response text (ReASC notebook reference metric) |

Labels never enter a policy's inputs; they are used only for training targets, dev selection and
evaluation.

```python
from pgm_asc.data.bank import load_bank
banks = load_bank("qwen7b", "math", "test")          # list of QuestionBank(question, records, vectors, labels)
import pyarrow.parquet as pq                          # or read the raw table directly
table = pq.read_table("artifacts/banks/qwen7b_math.parquet", columns=["split", "question_id", "prediction", "correct"])
```

## Top-20 log-probabilities (optional)

`banks/<model>_<dataset>.top_logprobs.parquet` has one row per response, in the same order as the bank:
`split`, `question_id`, `sample_index`, `top_k` and two flattened `[n_tokens × top_k]` arrays,
`top_token_ids` (int32) and `top_logprobs` (float32). They hold the log-probabilities vLLM returned at every
generated position: the top 20, plus the sampled token when it was outside them (`top_k` = 21); padding is
`-1` / `-inf`. They are the raw distribution, before temperature/top-p. Every file was checked when packing:
the sampled token appears exactly once per position with the bank's `sampled_logprobs` value, and the
`self_certainty` column is recomputed from them. Nothing in the study reads these files, so
`pgm download` skips them unless asked (`pgm download --only logprobs`).

```python
from pgm_asc.data.bank import load_top_logprobs
ids, lps = load_top_logprobs("qwen7b", "math", "test")[0]   # first response: arrays [n_tokens, top_k]
```

## Policies

`policies/<model>_<dataset>.json`:

| Key | Content |
|---|---|
| `representation` | Partition thresholds (cosine quantiles of train pairs) and hashes of the sampled pairs/questions |
| `scalers` | Train means/standard deviations of `x`, `w`, gate and candidate features |
| `heads.<arm>_s<seed>` | `spec`, `calibrator.T`, `selected` (dev-selected γ with its dev accuracy/tokens), `fixed95`, all dev `candidates`, `fit` (epochs, best epoch, dev NLL, learned α) |
| `dev_sc16_accuracy` | Reference for the selection constraint |
| `reasc_offline` | ReASC offline calibration (mean, std, gate) from the calibration cohort |
| `dev_tuned_baselines` | Dev-selected thresholds and candidates for ASC/ReASC |
| `converted_from` | SHA256 of the original selection lock (released policies were converted from the original run) |

`policies/<...>.checkpoints.pt` maps the same head names to `{state_dict, spec, embedding_dim, seed}`;
load with `torch.load(path, weights_only=True)` and rebuild with `pgm_asc.model.restore`.

## Decisions

One row per (method, question) with the stop index (`samples`), chosen answer, `correct`,
`reference_correct`, and the cost breakdown. The aggregate of these rows is
`results/evaluations/<cell>.json` in the GitHub repository.

## Provenance

The banks were produced by the original pipeline (GPU collection 2026-11-02 to 11-04) and repacked
without changing any value: every reported number is reproduced from them (see
[REPRODUCING.md](REPRODUCING.md#verification)). Not included: the duplicated pre-feature response files, the MATH confirm500 / previous-fresh banks that are
not part of the reported results, and the earlier V2/V5 experiments (code at git tag `v0-archive`; the
complete original archive is in `Ever2after/PGM-experiment-artifacts`).
