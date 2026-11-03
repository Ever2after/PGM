# Follow-up results

Aggregate results exported from the completed, locked evaluations. Primary training seed: 42; seeds 43/44 are stability checks. Omni-MATH remains deferred.

Output tokens include reasoning and both answer-extraction outputs. Input processing and the additional teacher-forcing (TF) pass are excluded from this token count. These numbers do not measure end-to-end serving latency.

## Main comparison

Each method uses the same frozen response bank within a model/dataset. `Selected` uses a dev-selected threshold; `fixed95` uses calibrated confidence 0.95. All learned arms have minimum 2 and cap 16 responses, without continuation.

| Model | Evaluation | N | SC16 acc. | ReASC offline acc. / tokens | Mixture selected acc. / tokens | Mixture fixed95 acc. / tokens |
|---|---|---:|---:|---:|---:|---:|
| Gemma-3-4B-IT | MATH500 | 500 | 77.40% | 76.80% / 5,648 | 77.80% / 4,571 | 78.40% / 7,243 |
| Gemma-3-4B-IT | MATH confirm500 | 500 | 75.60% | 75.60% / 5,100 | 77.00% / 4,392 | 76.60% / 7,340 |
| Gemma-3-4B-IT | GSM8K test | 1319 | 91.81% | 91.58% / 1,807 | 91.74% / 1,209 | 92.04% / 2,508 |
| Qwen2.5-3B-Instruct | MATH500 | 500 | 73.00% | 72.40% / 3,847 | 74.20% / 5,627 | 74.20% / 6,972 |
| Qwen2.5-3B-Instruct | MATH confirm500 | 500 | 72.60% | 69.80% / 4,157 | 73.80% / 5,862 | 74.00% / 7,143 |
| Qwen2.5-3B-Instruct | GSM8K test | 1319 | 90.45% | 89.01% / 1,012 | 88.70% / 775 | 89.92% / 1,340 |
| Qwen2.5-7B-Instruct | MATH500 | 500 | 79.80% | 78.00% / 3,112 | 79.20% / 2,949 | 80.60% / 5,382 |
| Qwen2.5-7B-Instruct | MATH confirm500 | 500 | 79.60% | 77.80% / 3,155 | 78.80% / 3,059 | 79.80% / 5,318 |
| Qwen2.5-7B-Instruct | GSM8K test | 1319 | 94.09% | 93.86% / 983 | 93.56% / 650 | 93.78% / 911 |

MATH500 is the standard OpenAI subset. Confirm500 is a separate original-MATH-test subset excluded from all earlier registered cohorts. `previous_fresh`, reported below for Gemma, is an earlier observed holdout and is exploratory. MATH train/dev use original train questions; the inherited 128-question ReASC calibration cohort contains 79 original-train and 49 original-test questions, disjoint from evaluation cohorts. GSM8K train/dev/calibration come from official train; evaluation uses all 1,319 official test questions.

No per-cell Holm-adjusted primary accuracy contrast currently establishes a semantic-mixture advantage over the four prespecified structure controls. This does not prove equivalence. Different dev-selected thresholds can change cost; fixed95 and fixed-prefix metrics provide additional comparisons.

The common judge is pinned xVerify-3B-Ia; the original ReASC alias was unavailable. This is a controlled reproduction, not a claim of identical historical benchmark scores.

## Complete per-evaluation tables

### gemma4b_math/math500

previously observed exploratory benchmark

| Method | Accuracy | Mean responses | Mean output tokens |
|---|---:|---:|---:|
| pass1 | 72.80% | 1.00 | 738 |
| sc16 | 77.40% | 16.00 | 11,852 |
| asc | 77.40% | 7.78 | 6,950 |
| esc | 77.40% | 8.06 | 7,219 |
| reasc_offline | 76.80% | 5.80 | 5,648 |
| reasc_online | 77.00% | 6.55 | 6,273 |
| asc_dev_tuned | 78.00% | 4.45 | 4,161 |
| reasc_dev_tuned | 77.20% | 3.77 | 3,686 |
| mixture / selected | 77.80% | 4.78 | 4,571 |
| mixture / fixed95 | 78.40% | 7.64 | 7,243 |
| uniform / selected | 77.80% | 4.86 | 4,658 |
| uniform / fixed95 | 78.40% | 7.69 | 7,261 |
| random / selected | 77.60% | 4.77 | 4,562 |
| random / fixed95 | 78.20% | 7.64 | 7,240 |
| alpha1 / selected | 78.00% | 5.30 | 5,097 |
| alpha1 / fixed95 | 78.40% | 7.65 | 7,193 |
| single_partition / selected | 77.20% | 4.99 | 4,783 |
| single_partition / fixed95 | 78.20% | 7.49 | 7,113 |
| all_one / selected | 77.80% | 5.05 | 4,847 |
| all_one / fixed95 | 78.20% | 7.62 | 7,181 |
| singletons / selected | 78.00% | 5.30 | 5,097 |
| singletons / fixed95 | 78.40% | 7.65 | 7,193 |
| candidate_counts / selected | 77.00% | 4.21 | 3,964 |
| candidate_counts / fixed95 | 77.40% | 8.63 | 7,720 |
| candidate_confidence / selected | 77.00% | 4.54 | 4,309 |
| candidate_confidence / fixed95 | 78.40% | 7.87 | 7,494 |
| deepsets_confidence / selected | 77.80% | 5.34 | 5,138 |
| deepsets_confidence / fixed95 | 78.00% | 8.19 | 7,635 |
| deepsets_embedding / selected | 77.60% | 5.40 | 5,200 |
| deepsets_embedding / fixed95 | 77.80% | 8.26 | 7,605 |
| same_head_single / selected | 78.20% | 6.01 | 5,824 |
| same_head_single / fixed95 | 78.40% | 8.61 | 7,886 |

Primary structure contrasts: mixture selected minus control selected, paired by question.

| Control | Accuracy delta (pp) | Paired 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.20 | [-1.20, +0.80] | 0.897 | 1.0000 |
| random | +0.20 | [+0.00, +0.60] | 1.002 | 1.0000 |
| candidate_confidence | +0.80 | [-0.40, +2.20] | 1.061 | 1.0000 |
| deepsets_embedding | +0.20 | [-1.20, +1.60] | 0.879 | 1.0000 |

### gemma4b_math/confirm

new confirmatory MATH holdout

| Method | Accuracy | Mean responses | Mean output tokens |
|---|---:|---:|---:|
| pass1 | 71.40% | 1.00 | 734 |
| sc16 | 75.60% | 16.00 | 11,794 |
| asc | 75.60% | 7.66 | 6,804 |
| esc | 75.60% | 8.08 | 7,194 |
| reasc_offline | 75.60% | 5.29 | 5,100 |
| reasc_online | 75.80% | 6.39 | 6,063 |
| asc_dev_tuned | 76.20% | 4.49 | 4,185 |
| reasc_dev_tuned | 75.60% | 3.47 | 3,346 |
| mixture / selected | 77.00% | 4.63 | 4,392 |
| mixture / fixed95 | 76.60% | 7.72 | 7,340 |
| uniform / selected | 76.80% | 4.70 | 4,464 |
| uniform / fixed95 | 76.80% | 7.71 | 7,315 |
| random / selected | 77.40% | 4.64 | 4,398 |
| random / fixed95 | 77.00% | 7.71 | 7,341 |
| alpha1 / selected | 77.60% | 5.02 | 4,784 |
| alpha1 / fixed95 | 76.60% | 7.62 | 7,183 |
| single_partition / selected | 76.60% | 4.76 | 4,531 |
| single_partition / fixed95 | 75.80% | 7.46 | 7,102 |
| all_one / selected | 77.20% | 4.82 | 4,580 |
| all_one / fixed95 | 76.40% | 7.58 | 7,187 |
| singletons / selected | 77.60% | 5.02 | 4,784 |
| singletons / fixed95 | 76.60% | 7.62 | 7,183 |
| candidate_counts / selected | 77.00% | 4.21 | 3,929 |
| candidate_counts / fixed95 | 75.60% | 8.68 | 7,706 |
| candidate_confidence / selected | 77.40% | 4.37 | 4,107 |
| candidate_confidence / fixed95 | 76.00% | 7.93 | 7,572 |
| deepsets_confidence / selected | 77.60% | 5.08 | 4,806 |
| deepsets_confidence / fixed95 | 77.20% | 8.17 | 7,635 |
| deepsets_embedding / selected | 76.40% | 5.14 | 4,908 |
| deepsets_embedding / fixed95 | 76.20% | 8.35 | 7,698 |
| same_head_single / selected | 76.40% | 5.91 | 5,702 |
| same_head_single / fixed95 | 76.20% | 8.81 | 8,144 |

Primary structure contrasts: mixture selected minus control selected, paired by question.

| Control | Accuracy delta (pp) | Paired 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.60 | [-1.60, +0.40] | 0.918 | 1.0000 |
| random | -0.40 | [-1.00, +0.00] | 0.999 | 1.0000 |
| candidate_confidence | -0.40 | [-1.60, +0.60] | 1.069 | 1.0000 |
| deepsets_embedding | +0.60 | [-0.80, +2.00] | 0.895 | 1.0000 |

### gemma4b_math/previous_fresh

previously observed exploratory benchmark

| Method | Accuracy | Mean responses | Mean output tokens |
|---|---:|---:|---:|
| pass1 | 73.80% | 1.00 | 723 |
| sc16 | 79.60% | 16.00 | 11,647 |
| asc | 79.60% | 7.35 | 6,473 |
| esc | 79.60% | 7.79 | 6,879 |
| reasc_offline | 78.80% | 5.29 | 5,096 |
| reasc_online | 79.60% | 6.15 | 5,830 |
| asc_dev_tuned | 80.00% | 4.37 | 4,070 |
| reasc_dev_tuned | 78.80% | 3.51 | 3,414 |
| mixture / selected | 81.00% | 4.72 | 4,496 |
| mixture / fixed95 | 80.80% | 7.24 | 6,795 |
| uniform / selected | 81.20% | 4.80 | 4,567 |
| uniform / fixed95 | 81.00% | 7.20 | 6,747 |
| random / selected | 80.60% | 4.71 | 4,486 |
| random / fixed95 | 80.40% | 7.23 | 6,792 |
| alpha1 / selected | 80.40% | 5.02 | 4,805 |
| alpha1 / fixed95 | 80.40% | 7.10 | 6,662 |
| single_partition / selected | 80.00% | 4.98 | 4,762 |
| single_partition / fixed95 | 79.80% | 7.07 | 6,629 |
| all_one / selected | 80.00% | 4.89 | 4,668 |
| all_one / fixed95 | 80.00% | 7.07 | 6,619 |
| singletons / selected | 80.40% | 5.02 | 4,805 |
| singletons / fixed95 | 80.40% | 7.10 | 6,662 |
| candidate_counts / selected | 79.40% | 4.27 | 4,011 |
| candidate_counts / fixed95 | 79.60% | 8.32 | 7,320 |
| candidate_confidence / selected | 79.60% | 4.43 | 4,173 |
| candidate_confidence / fixed95 | 80.00% | 7.27 | 6,889 |
| deepsets_confidence / selected | 80.80% | 5.12 | 4,889 |
| deepsets_confidence / fixed95 | 80.60% | 7.57 | 7,059 |
| deepsets_embedding / selected | 81.00% | 5.01 | 4,782 |
| deepsets_embedding / fixed95 | 80.80% | 7.94 | 7,266 |
| same_head_single / selected | 79.40% | 5.73 | 5,504 |
| same_head_single / fixed95 | 79.40% | 7.99 | 7,336 |

Primary structure contrasts: mixture selected minus control selected, paired by question.

| Control | Accuracy delta (pp) | Paired 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | +0.60 | [-0.20, +1.60] | 0.936 | 1.0000 |
| random | +0.40 | [+0.00, +1.00] | 1.002 | 1.0000 |
| candidate_confidence | +1.40 | [+0.00, +2.80] | 1.077 | 0.3691 |
| deepsets_embedding | +0.00 | [-1.20, +1.20] | 0.940 | 1.0000 |

### gemma4b_gsm8k/test

new model/dataset evaluation under frozen protocol

| Method | Accuracy | Mean responses | Mean output tokens |
|---|---:|---:|---:|
| pass1 | 89.92% | 1.00 | 445 |
| sc16 | 91.81% | 16.00 | 7,144 |
| asc | 91.81% | 5.10 | 2,651 |
| esc | 91.81% | 5.21 | 2,719 |
| reasc_offline | 91.58% | 2.88 | 1,807 |
| reasc_online | 91.66% | 3.83 | 2,229 |
| asc_dev_tuned | 91.74% | 2.51 | 1,325 |
| reasc_dev_tuned | 91.58% | 2.45 | 1,510 |
| mixture / selected | 91.74% | 2.34 | 1,209 |
| mixture / fixed95 | 92.04% | 4.09 | 2,508 |
| uniform / selected | 91.74% | 2.37 | 1,240 |
| uniform / fixed95 | 91.96% | 4.27 | 2,624 |
| random / selected | 91.66% | 2.35 | 1,218 |
| random / fixed95 | 92.12% | 4.08 | 2,508 |
| alpha1 / selected | 92.04% | 2.46 | 1,304 |
| alpha1 / fixed95 | 91.96% | 4.53 | 2,819 |
| single_partition / selected | 91.74% | 2.36 | 1,231 |
| single_partition / fixed95 | 92.12% | 4.13 | 2,558 |
| all_one / selected | 91.58% | 2.37 | 1,236 |
| all_one / fixed95 | 91.96% | 4.10 | 2,523 |
| singletons / selected | 92.04% | 2.46 | 1,304 |
| singletons / fixed95 | 91.96% | 4.53 | 2,819 |
| candidate_counts / selected | 91.74% | 2.58 | 1,386 |
| candidate_counts / fixed95 | 91.89% | 3.18 | 1,833 |
| candidate_confidence / selected | 91.43% | 2.51 | 1,344 |
| candidate_confidence / fixed95 | 91.81% | 4.15 | 2,551 |
| deepsets_confidence / selected | 91.28% | 2.27 | 1,142 |
| deepsets_confidence / fixed95 | 91.74% | 4.80 | 2,931 |
| deepsets_embedding / selected | 91.89% | 2.49 | 1,316 |
| deepsets_embedding / fixed95 | 91.66% | 5.05 | 3,008 |
| same_head_single / selected | 92.04% | 2.40 | 1,269 |
| same_head_single / fixed95 | 92.19% | 4.20 | 2,617 |

Primary structure contrasts: mixture selected minus control selected, paired by question.

| Control | Accuracy delta (pp) | Paired 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.30 | [-0.76, +0.15] | 0.927 | 1.0000 |
| random | +0.08 | [-0.15, +0.38] | 0.993 | 1.0000 |
| candidate_confidence | +0.30 | [-0.23, +0.83] | 0.900 | 1.0000 |
| deepsets_embedding | -0.15 | [-0.76, +0.45] | 0.919 | 1.0000 |

### qwen3b_math/math500

new model/dataset evaluation under frozen protocol

| Method | Accuracy | Mean responses | Mean output tokens |
|---|---:|---:|---:|
| pass1 | 66.00% | 1.00 | 598 |
| sc16 | 73.00% | 16.00 | 9,628 |
| asc | 73.20% | 8.99 | 6,278 |
| esc | 73.40% | 9.42 | 6,576 |
| reasc_offline | 72.40% | 5.41 | 3,847 |
| reasc_online | 73.80% | 7.28 | 5,201 |
| asc_dev_tuned | 73.40% | 7.75 | 5,586 |
| reasc_dev_tuned | 72.40% | 5.41 | 3,847 |
| mixture / selected | 74.20% | 7.58 | 5,627 |
| mixture / fixed95 | 74.20% | 9.99 | 6,972 |
| uniform / selected | 74.60% | 7.60 | 5,647 |
| uniform / fixed95 | 74.60% | 10.01 | 6,985 |
| random / selected | 74.20% | 7.56 | 5,613 |
| random / fixed95 | 74.20% | 10.00 | 6,982 |
| alpha1 / selected | 74.40% | 7.36 | 5,480 |
| alpha1 / fixed95 | 74.00% | 10.60 | 7,213 |
| single_partition / selected | 74.20% | 9.20 | 6,590 |
| single_partition / fixed95 | 74.00% | 9.89 | 6,925 |
| all_one / selected | 74.00% | 9.26 | 6,610 |
| all_one / fixed95 | 73.80% | 9.93 | 6,942 |
| singletons / selected | 74.40% | 7.36 | 5,480 |
| singletons / fixed95 | 74.00% | 10.60 | 7,213 |
| candidate_counts / selected | 73.60% | 6.73 | 5,045 |
| candidate_counts / fixed95 | 73.20% | 10.06 | 6,950 |
| candidate_confidence / selected | 73.80% | 7.71 | 5,698 |
| candidate_confidence / fixed95 | 73.40% | 10.77 | 7,435 |
| deepsets_confidence / selected | 72.60% | 5.70 | 4,359 |
| deepsets_confidence / fixed95 | 73.00% | 11.04 | 7,444 |
| deepsets_embedding / selected | 73.80% | 7.16 | 5,376 |
| deepsets_embedding / fixed95 | 73.60% | 10.03 | 7,004 |
| same_head_single / selected | 74.00% | 8.71 | 6,320 |
| same_head_single / fixed95 | 73.80% | 10.87 | 7,408 |

Primary structure contrasts: mixture selected minus control selected, paired by question.

| Control | Accuracy delta (pp) | Paired 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.20 | [-1.20, +0.60] | 1.027 | 1.0000 |
| random | +0.00 | [+0.00, +0.00] | 1.003 | 1.0000 |
| candidate_confidence | +0.40 | [-0.80, +1.60] | 0.988 | 1.0000 |
| deepsets_embedding | +0.40 | [-0.80, +1.60] | 1.047 | 1.0000 |

### qwen3b_math/confirm

new confirmatory MATH holdout

| Method | Accuracy | Mean responses | Mean output tokens |
|---|---:|---:|---:|
| pass1 | 61.80% | 1.00 | 610 |
| sc16 | 72.60% | 16.00 | 9,670 |
| asc | 72.60% | 9.04 | 6,428 |
| esc | 72.60% | 9.40 | 6,688 |
| reasc_offline | 69.80% | 5.83 | 4,157 |
| reasc_online | 72.00% | 7.70 | 5,689 |
| asc_dev_tuned | 72.60% | 7.83 | 5,771 |
| reasc_dev_tuned | 69.80% | 5.83 | 4,157 |
| mixture / selected | 73.80% | 7.68 | 5,862 |
| mixture / fixed95 | 74.00% | 10.06 | 7,143 |
| uniform / selected | 73.60% | 7.69 | 5,869 |
| uniform / fixed95 | 73.80% | 10.07 | 7,143 |
| random / selected | 73.80% | 7.68 | 5,869 |
| random / fixed95 | 74.00% | 10.07 | 7,134 |
| alpha1 / selected | 73.80% | 7.45 | 5,711 |
| alpha1 / fixed95 | 74.20% | 10.55 | 7,321 |
| single_partition / selected | 73.80% | 9.31 | 6,772 |
| single_partition / fixed95 | 73.80% | 10.08 | 7,151 |
| all_one / selected | 74.20% | 9.32 | 6,771 |
| all_one / fixed95 | 74.20% | 10.06 | 7,157 |
| singletons / selected | 73.80% | 7.45 | 5,711 |
| singletons / fixed95 | 74.20% | 10.55 | 7,321 |
| candidate_counts / selected | 72.00% | 6.84 | 5,315 |
| candidate_counts / fixed95 | 72.60% | 10.15 | 7,119 |
| candidate_confidence / selected | 73.20% | 7.70 | 5,884 |
| candidate_confidence / fixed95 | 73.80% | 10.73 | 7,553 |
| deepsets_confidence / selected | 73.20% | 5.97 | 4,695 |
| deepsets_confidence / fixed95 | 74.20% | 11.08 | 7,625 |
| deepsets_embedding / selected | 73.20% | 7.17 | 5,532 |
| deepsets_embedding / fixed95 | 73.20% | 10.19 | 7,213 |
| same_head_single / selected | 73.60% | 8.89 | 6,561 |
| same_head_single / fixed95 | 73.80% | 10.77 | 7,497 |

Primary structure contrasts: mixture selected minus control selected, paired by question.

| Control | Accuracy delta (pp) | Paired 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | +0.00 | [-0.60, +0.60] | 1.026 | 1.0000 |
| random | +0.00 | [+0.00, +0.00] | 0.999 | 1.0000 |
| candidate_confidence | +0.60 | [-0.60, +2.00] | 0.996 | 1.0000 |
| deepsets_embedding | +0.60 | [-0.80, +2.20] | 1.060 | 1.0000 |

### qwen3b_gsm8k/test

new model/dataset evaluation under frozen protocol

| Method | Accuracy | Mean responses | Mean output tokens |
|---|---:|---:|---:|
| pass1 | 85.29% | 1.00 | 302 |
| sc16 | 90.45% | 16.00 | 4,889 |
| asc | 90.45% | 5.85 | 1,912 |
| esc | 90.45% | 6.11 | 2,010 |
| reasc_offline | 89.01% | 2.96 | 1,012 |
| reasc_online | 90.45% | 4.87 | 1,623 |
| asc_dev_tuned | 85.29% | 1.00 | 302 |
| reasc_dev_tuned | 85.29% | 1.00 | 302 |
| mixture / selected | 88.70% | 2.38 | 775 |
| mixture / fixed95 | 89.92% | 3.87 | 1,340 |
| uniform / selected | 88.63% | 2.40 | 785 |
| uniform / fixed95 | 89.69% | 3.92 | 1,361 |
| random / selected | 88.78% | 2.39 | 778 |
| random / fixed95 | 89.99% | 3.91 | 1,357 |
| alpha1 / selected | 89.16% | 2.50 | 825 |
| alpha1 / fixed95 | 89.99% | 4.05 | 1,405 |
| single_partition / selected | 88.63% | 2.43 | 797 |
| single_partition / fixed95 | 89.84% | 3.98 | 1,373 |
| all_one / selected | 88.93% | 2.46 | 802 |
| all_one / fixed95 | 90.07% | 3.92 | 1,345 |
| singletons / selected | 89.16% | 2.50 | 825 |
| singletons / fixed95 | 89.99% | 4.05 | 1,405 |
| candidate_counts / selected | 88.70% | 2.33 | 741 |
| candidate_counts / fixed95 | 90.14% | 3.61 | 1,220 |
| candidate_confidence / selected | 89.54% | 2.93 | 987 |
| candidate_confidence / fixed95 | 89.99% | 3.93 | 1,357 |
| deepsets_confidence / selected | 88.70% | 2.38 | 773 |
| deepsets_confidence / fixed95 | 89.84% | 4.45 | 1,547 |
| deepsets_embedding / selected | 89.08% | 2.59 | 864 |
| deepsets_embedding / fixed95 | 90.14% | 4.63 | 1,604 |
| same_head_single / selected | 88.63% | 2.49 | 822 |
| same_head_single / fixed95 | 89.84% | 4.15 | 1,444 |

Primary structure contrasts: mixture selected minus control selected, paired by question.

| Control | Accuracy delta (pp) | Paired 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.45 | [-0.99, +0.08] | 0.940 | 0.4380 |
| random | -0.08 | [-0.23, +0.00] | 0.996 | 1.0000 |
| candidate_confidence | -0.83 | [-1.74, +0.08] | 0.785 | 0.3955 |
| deepsets_embedding | -0.38 | [-1.21, +0.45] | 0.898 | 0.9737 |

### qwen7b_math/math500

new model/dataset evaluation under frozen protocol

| Method | Accuracy | Mean responses | Mean output tokens |
|---|---:|---:|---:|
| pass1 | 74.60% | 1.00 | 567 |
| sc16 | 79.80% | 16.00 | 9,111 |
| asc | 79.80% | 7.30 | 4,894 |
| esc | 79.80% | 7.70 | 5,176 |
| reasc_offline | 78.00% | 4.46 | 3,112 |
| reasc_online | 79.60% | 5.73 | 3,993 |
| asc_dev_tuned | 79.60% | 3.90 | 2,789 |
| reasc_dev_tuned | 78.00% | 3.17 | 2,285 |
| mixture / selected | 79.20% | 4.06 | 2,949 |
| mixture / fixed95 | 80.60% | 7.77 | 5,382 |
| uniform / selected | 79.80% | 4.21 | 3,056 |
| uniform / fixed95 | 80.40% | 7.88 | 5,450 |
| random / selected | 79.00% | 4.06 | 2,953 |
| random / fixed95 | 80.40% | 7.76 | 5,370 |
| alpha1 / selected | 79.80% | 4.43 | 3,200 |
| alpha1 / fixed95 | 80.60% | 8.28 | 5,630 |
| single_partition / selected | 79.80% | 4.30 | 3,113 |
| single_partition / fixed95 | 80.80% | 7.79 | 5,387 |
| all_one / selected | 79.00% | 4.23 | 3,076 |
| all_one / fixed95 | 80.40% | 7.79 | 5,388 |
| singletons / selected | 79.80% | 4.43 | 3,200 |
| singletons / fixed95 | 80.60% | 8.28 | 5,630 |
| candidate_counts / selected | 77.60% | 3.66 | 2,684 |
| candidate_counts / fixed95 | 79.80% | 8.27 | 5,513 |
| candidate_confidence / selected | 79.20% | 4.06 | 2,964 |
| candidate_confidence / fixed95 | 81.40% | 8.49 | 5,853 |
| deepsets_confidence / selected | 79.40% | 4.30 | 3,109 |
| deepsets_confidence / fixed95 | 80.00% | 9.13 | 6,054 |
| deepsets_embedding / selected | 79.20% | 4.59 | 3,354 |
| deepsets_embedding / fixed95 | 80.20% | 8.88 | 6,061 |
| same_head_single / selected | 79.80% | 4.59 | 3,339 |
| same_head_single / fixed95 | 80.60% | 8.58 | 5,832 |

Primary structure contrasts: mixture selected minus control selected, paired by question.

| Control | Accuracy delta (pp) | Paired 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.60 | [-1.60, +0.40] | 0.922 | 1.0000 |
| random | +0.20 | [+0.00, +0.60] | 0.999 | 1.0000 |
| candidate_confidence | +0.00 | [-1.00, +1.20] | 0.995 | 1.0000 |
| deepsets_embedding | +0.00 | [-1.00, +1.00] | 0.879 | 1.0000 |

### qwen7b_math/confirm

new confirmatory MATH holdout

| Method | Accuracy | Mean responses | Mean output tokens |
|---|---:|---:|---:|
| pass1 | 73.40% | 1.00 | 568 |
| sc16 | 79.60% | 16.00 | 9,112 |
| asc | 79.60% | 7.33 | 5,018 |
| esc | 79.80% | 7.62 | 5,224 |
| reasc_offline | 77.80% | 4.50 | 3,155 |
| reasc_online | 79.00% | 5.87 | 4,235 |
| asc_dev_tuned | 78.80% | 4.03 | 2,937 |
| reasc_dev_tuned | 77.40% | 3.22 | 2,318 |
| mixture / selected | 78.80% | 4.15 | 3,059 |
| mixture / fixed95 | 79.80% | 7.46 | 5,318 |
| uniform / selected | 79.00% | 4.26 | 3,150 |
| uniform / fixed95 | 79.80% | 7.56 | 5,364 |
| random / selected | 78.60% | 4.12 | 3,027 |
| random / fixed95 | 79.60% | 7.49 | 5,327 |
| alpha1 / selected | 79.20% | 4.65 | 3,465 |
| alpha1 / fixed95 | 80.00% | 8.03 | 5,580 |
| single_partition / selected | 78.80% | 4.46 | 3,303 |
| single_partition / fixed95 | 79.40% | 7.46 | 5,307 |
| all_one / selected | 79.00% | 4.44 | 3,300 |
| all_one / fixed95 | 80.00% | 7.46 | 5,308 |
| singletons / selected | 79.20% | 4.65 | 3,465 |
| singletons / fixed95 | 80.00% | 8.03 | 5,580 |
| candidate_counts / selected | 78.20% | 3.88 | 2,864 |
| candidate_counts / fixed95 | 79.60% | 8.18 | 5,633 |
| candidate_confidence / selected | 78.80% | 4.12 | 3,039 |
| candidate_confidence / fixed95 | 80.00% | 8.19 | 5,801 |
| deepsets_confidence / selected | 78.60% | 4.44 | 3,262 |
| deepsets_confidence / fixed95 | 78.80% | 8.97 | 6,007 |
| deepsets_embedding / selected | 79.00% | 4.57 | 3,420 |
| deepsets_embedding / fixed95 | 79.60% | 8.55 | 5,961 |
| same_head_single / selected | 79.20% | 4.85 | 3,618 |
| same_head_single / fixed95 | 79.60% | 8.28 | 5,764 |

Primary structure contrasts: mixture selected minus control selected, paired by question.

| Control | Accuracy delta (pp) | Paired 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.40 | [-1.60, +0.80] | 0.883 | 1.0000 |
| random | +0.20 | [+0.00, +0.60] | 1.010 | 1.0000 |
| candidate_confidence | +0.00 | [-1.00, +1.00] | 1.007 | 1.0000 |
| deepsets_embedding | -0.20 | [-1.20, +0.80] | 0.894 | 1.0000 |

### qwen7b_gsm8k/test

new model/dataset evaluation under frozen protocol

| Method | Accuracy | Mean responses | Mean output tokens |
|---|---:|---:|---:|
| pass1 | 92.12% | 1.00 | 289 |
| sc16 | 94.09% | 16.00 | 4,635 |
| asc | 94.09% | 4.91 | 1,494 |
| esc | 94.09% | 5.09 | 1,559 |
| reasc_offline | 93.86% | 2.99 | 983 |
| reasc_online | 94.16% | 4.10 | 1,294 |
| asc_dev_tuned | 93.78% | 2.33 | 707 |
| reasc_dev_tuned | 93.63% | 1.71 | 541 |
| mixture / selected | 93.56% | 2.18 | 650 |
| mixture / fixed95 | 93.78% | 2.85 | 911 |
| uniform / selected | 93.56% | 2.22 | 668 |
| uniform / fixed95 | 93.71% | 2.93 | 940 |
| random / selected | 93.48% | 2.18 | 651 |
| random / fixed95 | 93.78% | 2.83 | 904 |
| alpha1 / selected | 93.56% | 2.27 | 692 |
| alpha1 / fixed95 | 93.48% | 2.94 | 952 |
| single_partition / selected | 93.40% | 2.22 | 669 |
| single_partition / fixed95 | 93.71% | 2.84 | 905 |
| all_one / selected | 93.40% | 2.20 | 660 |
| all_one / fixed95 | 93.63% | 2.85 | 911 |
| singletons / selected | 93.56% | 2.27 | 692 |
| singletons / fixed95 | 93.48% | 2.94 | 952 |
| candidate_counts / selected | 93.48% | 2.13 | 631 |
| candidate_counts / fixed95 | 93.63% | 2.59 | 804 |
| candidate_confidence / selected | 93.40% | 2.16 | 646 |
| candidate_confidence / fixed95 | 93.71% | 2.86 | 914 |
| deepsets_confidence / selected | 93.63% | 2.17 | 648 |
| deepsets_confidence / fixed95 | 93.71% | 2.74 | 859 |
| deepsets_embedding / selected | 93.56% | 2.30 | 703 |
| deepsets_embedding / fixed95 | 93.40% | 2.95 | 955 |
| same_head_single / selected | 93.48% | 2.20 | 662 |
| same_head_single / fixed95 | 93.71% | 2.88 | 922 |

Primary structure contrasts: mixture selected minus control selected, paired by question.

| Control | Accuracy delta (pp) | Paired 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | +0.00 | [-0.38, +0.38] | 0.940 | 1.0000 |
| random | +0.08 | [+0.00, +0.23] | 0.998 | 1.0000 |
| candidate_confidence | +0.15 | [-0.15, +0.45] | 1.007 | 1.0000 |
| deepsets_embedding | +0.00 | [-0.38, +0.38] | 0.925 | 1.0000 |

## Machine-readable results

[metrics.json](metrics.json) retains all seeds, common operating points, fixed-prefix metrics, primary/secondary contrasts, source hashes and cost caveats. [provenance/](provenance/) retains cohort identities and original locks. Original artifact paths in those records refer to the local artifact bundle, which is not included in Git.
