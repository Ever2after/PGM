# Results

Primary training seed 42 (seeds 43/44 are in the JSON files). Each entry is **accuracy / mean output tokens per question**; output tokens = generated tokens + both answer-extraction outputs (input processing and the teacher-forcing pass are excluded). PGM mixture and ASC dev-tuned use thresholds chosen on dev (fewest tokens with dev accuracy ≥ dev SC16 − 1 pp).

## In-domain

Policies are fitted on the same dataset (calibration/train/dev = 128/800/200 questions).

| Model | Evaluation | N | SC16 | ReASC offline | ASC dev-tuned | PGM mixture | PGM mixture, γ=0.95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma-3-4B-IT | MATH500 | 500 | 77.40% / 11,852 | 76.80% / 5,648 | 78.00% / 4,161 | 77.80% / 4,571 | 78.40% / 7,243 |
| Gemma-3-4B-IT | GSM8K | 1319 | 91.81% / 7,144 | 91.58% / 1,807 | 91.74% / 1,325 | 91.74% / 1,209 | 92.04% / 2,508 |
| Gemma-3-4B-IT | Omni-MATH | 999 | 31.63% / 21,542 | 31.73% / 14,346 | 31.93% / 11,047 | 32.23% / 19,020 | 32.13% / 21,317 |
| Qwen2.5-3B-Instruct | MATH500 | 500 | 73.00% / 9,628 | 72.40% / 3,847 | 73.40% / 5,586 | 74.20% / 5,627 | 74.20% / 6,972 |
| Qwen2.5-3B-Instruct | GSM8K | 1319 | 90.45% / 4,889 | 89.01% / 1,012 | 85.29% / 302 | 88.70% / 775 | 89.92% / 1,340 |
| Qwen2.5-3B-Instruct | Omni-MATH | 999 | 28.03% / 13,221 | 27.43% / 9,296 | 27.43% / 7,529 | 27.93% / 11,385 | 27.83% / 12,894 |
| Qwen2.5-7B-Instruct | MATH500 | 500 | 79.80% / 9,111 | 78.00% / 3,112 | 79.60% / 2,789 | 79.20% / 2,949 | 80.60% / 5,382 |
| Qwen2.5-7B-Instruct | GSM8K | 1319 | 94.09% / 4,635 | 93.86% / 983 | 93.78% / 707 | 93.56% / 650 | 93.78% / 911 |
| Qwen2.5-7B-Instruct | Omni-MATH | 999 | 33.53% / 13,156 | 32.53% / 8,335 | 33.33% / 6,618 | 34.13% / 10,715 | 34.13% / 12,850 |
| Llama-3.1-8B-Instruct | MATH500 | 500 | 58.80% / 7,709 | 58.40% / 4,311 | 59.60% / 3,997 | 60.80% / 4,062 | 60.60% / 5,963 |
| Llama-3.1-8B-Instruct | GSM8K | 1319 | 90.37% / 2,885 | 88.93% / 751 | 89.54% / 609 | 89.39% / 600 | 90.14% / 785 |
| Llama-3.1-8B-Instruct | Omni-MATH | 999 | 17.22% / 16,941 | 17.92% / 14,409 | 17.02% / 12,171 | 17.42% / 15,988 | 17.62% / 16,820 |

## Transfer: MATH → Omni-MATH

Each model's MATH policy (heads, calibrators, thresholds, ReASC offline calibration) is replayed unchanged on the Omni-MATH test bank, without Omni labels.

| Model | Evaluation | N | SC16 | ReASC offline | ASC dev-tuned | PGM mixture | PGM mixture, γ=0.95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma-3-4B-IT | MATH → Omni-MATH | 999 | 31.63% / 21,542 | 31.33% / 15,277 | 31.93% / 9,456 | 30.93% / 7,576 | 31.43% / 16,025 |
| Qwen2.5-3B-Instruct | MATH → Omni-MATH | 999 | 28.03% / 13,221 | 27.63% / 9,743 | 28.03% / 10,272 | 28.33% / 11,785 | 28.23% / 12,588 |
| Qwen2.5-7B-Instruct | MATH → Omni-MATH | 999 | 33.53% / 13,156 | 32.73% / 9,114 | 33.33% / 6,618 | 33.03% / 7,390 | 33.73% / 11,795 |
| Llama-3.1-8B-Instruct | MATH → Omni-MATH | 999 | 17.22% / 16,941 | 17.12% / 14,986 | 17.02% / 12,171 | 17.12% / 14,200 | 17.32% / 16,477 |

## Structure contrasts

1 of 48 in-domain primary contrasts favor mixture at per-cell Holm p < 0.05 (llama8b_gsm8k: mixture vs candidate_confidence).
Different dev-selected thresholds change cost, so accuracy contrasts are not equal-budget comparisons.

## All methods and ablations

### Gemma-3-4B-IT, MATH500

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 72.80% | 1.00 | 738 | | | |
| sc16 | 77.40% | 16.00 | 11,852 | | | |
| asc | 77.40% | 7.78 | 6,950 | | | |
| esc | 77.40% | 8.06 | 7,219 | | | |
| reasc_offline | 76.80% | 5.80 | 5,648 | | | |
| reasc_online | 77.00% | 6.55 | 6,273 | | | |
| asc_dev_tuned | 78.00% | 4.45 | 4,161 | | | |
| reasc_dev_tuned | 77.20% | 3.77 | 3,686 | | | |
| mixture | 77.80% | 4.78 | 4,571 | 78.40% | 7.64 | 7,243 |
| uniform | 77.80% | 4.86 | 4,658 | 78.40% | 7.69 | 7,261 |
| random | 77.60% | 4.77 | 4,562 | 78.20% | 7.64 | 7,240 |
| alpha1 | 78.00% | 5.30 | 5,097 | 78.40% | 7.65 | 7,193 |
| single_partition | 77.20% | 4.99 | 4,783 | 78.20% | 7.49 | 7,113 |
| all_one | 77.80% | 5.05 | 4,847 | 78.20% | 7.62 | 7,181 |
| singletons | 78.00% | 5.30 | 5,097 | 78.40% | 7.65 | 7,193 |
| candidate_counts | 77.00% | 4.21 | 3,964 | 77.40% | 8.63 | 7,720 |
| candidate_confidence | 77.00% | 4.54 | 4,309 | 78.40% | 7.87 | 7,494 |
| deepsets_confidence | 77.80% | 5.34 | 5,138 | 78.00% | 8.19 | 7,635 |
| deepsets_embedding | 77.60% | 5.40 | 5,200 | 77.80% | 8.26 | 7,605 |
| same_head_single | 78.20% | 6.01 | 5,824 | 78.40% | 8.61 | 7,886 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.20 | [-1.20, +0.80] | 0.897 | 1.0000 |
| random | +0.20 | [+0.00, +0.60] | 1.002 | 1.0000 |
| candidate_confidence | +0.80 | [-0.40, +2.20] | 1.061 | 1.0000 |
| deepsets_embedding | +0.20 | [-1.20, +1.60] | 0.879 | 1.0000 |

### Gemma-3-4B-IT, GSM8K

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 89.92% | 1.00 | 445 | | | |
| sc16 | 91.81% | 16.00 | 7,144 | | | |
| asc | 91.81% | 5.10 | 2,651 | | | |
| esc | 91.81% | 5.21 | 2,719 | | | |
| reasc_offline | 91.58% | 2.88 | 1,807 | | | |
| reasc_online | 91.66% | 3.83 | 2,229 | | | |
| asc_dev_tuned | 91.74% | 2.51 | 1,325 | | | |
| reasc_dev_tuned | 91.58% | 2.45 | 1,510 | | | |
| mixture | 91.74% | 2.34 | 1,209 | 92.04% | 4.09 | 2,508 |
| uniform | 91.74% | 2.37 | 1,240 | 91.96% | 4.27 | 2,624 |
| random | 91.66% | 2.35 | 1,218 | 92.12% | 4.08 | 2,508 |
| alpha1 | 92.04% | 2.46 | 1,304 | 91.96% | 4.53 | 2,819 |
| single_partition | 91.74% | 2.36 | 1,231 | 92.12% | 4.13 | 2,558 |
| all_one | 91.58% | 2.37 | 1,236 | 91.96% | 4.10 | 2,523 |
| singletons | 92.04% | 2.46 | 1,304 | 91.96% | 4.53 | 2,819 |
| candidate_counts | 91.74% | 2.58 | 1,386 | 91.89% | 3.18 | 1,833 |
| candidate_confidence | 91.43% | 2.51 | 1,344 | 91.81% | 4.15 | 2,551 |
| deepsets_confidence | 91.28% | 2.27 | 1,142 | 91.74% | 4.80 | 2,931 |
| deepsets_embedding | 91.89% | 2.49 | 1,316 | 91.66% | 5.05 | 3,008 |
| same_head_single | 92.04% | 2.40 | 1,269 | 92.19% | 4.20 | 2,617 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.30 | [-0.76, +0.15] | 0.927 | 1.0000 |
| random | +0.08 | [-0.15, +0.38] | 0.993 | 1.0000 |
| candidate_confidence | +0.30 | [-0.23, +0.83] | 0.900 | 1.0000 |
| deepsets_embedding | -0.15 | [-0.76, +0.45] | 0.919 | 1.0000 |

### Gemma-3-4B-IT, Omni-MATH

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 26.33% | 1.00 | 1,345 | | | |
| sc16 | 31.63% | 16.00 | 21,542 | | | |
| asc | 31.63% | 11.04 | 15,898 | | | |
| esc | 31.63% | 11.79 | 16,983 | | | |
| reasc_offline | 31.73% | 9.77 | 14,346 | | | |
| reasc_online | 31.53% | 9.81 | 14,417 | | | |
| asc_dev_tuned | 31.93% | 7.45 | 11,047 | | | |
| reasc_dev_tuned | 31.43% | 8.21 | 12,137 | | | |
| mixture | 32.23% | 13.18 | 19,020 | 32.13% | 15.31 | 21,317 |
| uniform | 32.03% | 13.08 | 18,863 | 32.13% | 15.30 | 21,321 |
| random | 32.23% | 13.12 | 18,923 | 32.23% | 15.30 | 21,316 |
| alpha1 | 31.83% | 13.18 | 19,009 | 31.83% | 15.37 | 21,346 |
| single_partition | 31.93% | 13.08 | 18,878 | 32.03% | 15.28 | 21,290 |
| all_one | 32.23% | 13.25 | 19,137 | 32.13% | 15.35 | 21,326 |
| singletons | 31.83% | 13.18 | 19,009 | 31.83% | 15.37 | 21,346 |
| candidate_counts | 31.53% | 15.46 | 21,087 | 31.53% | 16.00 | 21,542 |
| candidate_confidence | 32.53% | 13.41 | 19,359 | 32.53% | 15.51 | 21,392 |
| deepsets_confidence | 32.33% | 13.17 | 18,971 | 32.33% | 15.27 | 21,302 |
| deepsets_embedding | 32.13% | 12.97 | 18,820 | 32.03% | 14.85 | 21,004 |
| same_head_single | 32.03% | 13.18 | 19,076 | 32.03% | 15.31 | 21,314 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | +0.40 | [-0.10, +1.00] | 1.001 | 1.0000 |
| random | +0.00 | [+0.00, +0.00] | 1.005 | 1.0000 |
| candidate_confidence | -0.30 | [-1.40, +0.80] | 0.983 | 1.0000 |
| deepsets_embedding | +0.10 | [-0.60, +0.80] | 1.011 | 1.0000 |

### Gemma-3-4B-IT, MATH → Omni-MATH

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 26.33% | 1.00 | 1,345 | | | |
| sc16 | 31.63% | 16.00 | 21,542 | | | |
| asc | 31.63% | 11.04 | 15,898 | | | |
| esc | 31.63% | 11.79 | 16,983 | | | |
| reasc_offline | 31.33% | 10.41 | 15,277 | | | |
| reasc_online | 31.53% | 9.81 | 14,417 | | | |
| asc_dev_tuned | 31.93% | 6.36 | 9,456 | | | |
| reasc_dev_tuned | 31.43% | 6.09 | 9,172 | | | |
| mixture | 30.93% | 5.28 | 7,576 | 31.43% | 11.13 | 16,025 |
| uniform | 30.73% | 5.78 | 8,324 | 31.43% | 11.17 | 16,097 |
| random | 30.93% | 5.29 | 7,582 | 31.43% | 11.14 | 16,054 |
| alpha1 | 31.03% | 6.31 | 9,113 | 31.83% | 10.69 | 15,404 |
| single_partition | 30.13% | 5.79 | 8,411 | 31.53% | 10.53 | 15,225 |
| all_one | 30.73% | 5.89 | 8,453 | 31.43% | 10.46 | 15,056 |
| singletons | 31.03% | 6.31 | 9,113 | 31.83% | 10.69 | 15,404 |
| candidate_counts | 31.43% | 5.67 | 8,533 | 31.63% | 12.55 | 18,043 |
| candidate_confidence | 30.93% | 5.29 | 7,774 | 31.43% | 12.07 | 17,446 |
| deepsets_confidence | 31.93% | 6.67 | 9,728 | 32.13% | 11.73 | 16,946 |
| deepsets_embedding | 32.03% | 7.94 | 11,564 | 32.23% | 13.11 | 18,861 |
| same_head_single | 30.83% | 8.28 | 12,013 | 31.23% | 11.99 | 17,269 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.10 | [-0.70, +0.50] | 0.831 | 1.0000 |
| random | +0.00 | [+0.00, +0.00] | 0.999 | 1.0000 |
| candidate_confidence | +0.00 | [-0.90, +0.90] | 0.975 | 1.0000 |
| deepsets_embedding | -1.10 | [-2.10, -0.10] | 0.655 | 0.2090 |

### Qwen2.5-3B-Instruct, MATH500

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 66.00% | 1.00 | 598 | | | |
| sc16 | 73.00% | 16.00 | 9,628 | | | |
| asc | 73.20% | 8.99 | 6,278 | | | |
| esc | 73.40% | 9.42 | 6,576 | | | |
| reasc_offline | 72.40% | 5.41 | 3,847 | | | |
| reasc_online | 73.80% | 7.28 | 5,201 | | | |
| asc_dev_tuned | 73.40% | 7.75 | 5,586 | | | |
| reasc_dev_tuned | 72.40% | 5.41 | 3,847 | | | |
| mixture | 74.20% | 7.58 | 5,627 | 74.20% | 9.99 | 6,972 |
| uniform | 74.60% | 7.60 | 5,647 | 74.60% | 10.01 | 6,985 |
| random | 74.20% | 7.56 | 5,613 | 74.20% | 10.00 | 6,982 |
| alpha1 | 74.40% | 7.36 | 5,480 | 74.00% | 10.60 | 7,213 |
| single_partition | 74.20% | 9.20 | 6,590 | 74.00% | 9.89 | 6,925 |
| all_one | 74.00% | 9.26 | 6,610 | 73.80% | 9.93 | 6,942 |
| singletons | 74.40% | 7.36 | 5,480 | 74.00% | 10.60 | 7,213 |
| candidate_counts | 73.60% | 6.73 | 5,045 | 73.20% | 10.06 | 6,950 |
| candidate_confidence | 73.80% | 7.71 | 5,698 | 73.40% | 10.77 | 7,435 |
| deepsets_confidence | 72.60% | 5.70 | 4,359 | 73.00% | 11.04 | 7,444 |
| deepsets_embedding | 73.80% | 7.16 | 5,376 | 73.60% | 10.03 | 7,004 |
| same_head_single | 74.00% | 8.71 | 6,320 | 73.80% | 10.87 | 7,408 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.20 | [-1.20, +0.60] | 1.027 | 1.0000 |
| random | +0.00 | [+0.00, +0.00] | 1.003 | 1.0000 |
| candidate_confidence | +0.40 | [-0.80, +1.60] | 0.988 | 1.0000 |
| deepsets_embedding | +0.40 | [-0.80, +1.60] | 1.047 | 1.0000 |

### Qwen2.5-3B-Instruct, GSM8K

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 85.29% | 1.00 | 302 | | | |
| sc16 | 90.45% | 16.00 | 4,889 | | | |
| asc | 90.45% | 5.85 | 1,912 | | | |
| esc | 90.45% | 6.11 | 2,010 | | | |
| reasc_offline | 89.01% | 2.96 | 1,012 | | | |
| reasc_online | 90.45% | 4.87 | 1,623 | | | |
| asc_dev_tuned | 85.29% | 1.00 | 302 | | | |
| reasc_dev_tuned | 85.29% | 1.00 | 302 | | | |
| mixture | 88.70% | 2.38 | 775 | 89.92% | 3.87 | 1,340 |
| uniform | 88.63% | 2.40 | 785 | 89.69% | 3.92 | 1,361 |
| random | 88.78% | 2.39 | 778 | 89.99% | 3.91 | 1,357 |
| alpha1 | 89.16% | 2.50 | 825 | 89.99% | 4.05 | 1,405 |
| single_partition | 88.63% | 2.43 | 797 | 89.84% | 3.98 | 1,373 |
| all_one | 88.93% | 2.46 | 802 | 90.07% | 3.92 | 1,345 |
| singletons | 89.16% | 2.50 | 825 | 89.99% | 4.05 | 1,405 |
| candidate_counts | 88.70% | 2.33 | 741 | 90.14% | 3.61 | 1,220 |
| candidate_confidence | 89.54% | 2.93 | 987 | 89.99% | 3.93 | 1,357 |
| deepsets_confidence | 88.70% | 2.38 | 773 | 89.84% | 4.45 | 1,547 |
| deepsets_embedding | 89.08% | 2.59 | 864 | 90.14% | 4.63 | 1,604 |
| same_head_single | 88.63% | 2.49 | 822 | 89.84% | 4.15 | 1,444 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.45 | [-0.99, +0.08] | 0.940 | 0.4380 |
| random | -0.08 | [-0.23, +0.00] | 0.996 | 1.0000 |
| candidate_confidence | -0.83 | [-1.74, +0.08] | 0.785 | 0.3955 |
| deepsets_embedding | -0.38 | [-1.21, +0.45] | 0.898 | 0.9737 |

### Qwen2.5-3B-Instruct, Omni-MATH

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 23.32% | 1.00 | 833 | | | |
| sc16 | 28.03% | 16.00 | 13,221 | | | |
| asc | 28.03% | 12.78 | 11,174 | | | |
| esc | 28.03% | 13.39 | 11,686 | | | |
| reasc_offline | 27.43% | 10.71 | 9,296 | | | |
| reasc_online | 27.63% | 11.08 | 9,601 | | | |
| asc_dev_tuned | 27.43% | 8.39 | 7,529 | | | |
| reasc_dev_tuned | 25.23% | 5.70 | 4,723 | | | |
| mixture | 27.93% | 13.00 | 11,385 | 27.83% | 15.13 | 12,894 |
| uniform | 27.23% | 13.48 | 11,774 | 27.13% | 15.19 | 12,921 |
| random | 27.73% | 13.00 | 11,368 | 27.73% | 15.13 | 12,892 |
| alpha1 | 27.73% | 13.25 | 11,586 | 27.63% | 15.06 | 12,845 |
| single_partition | 28.33% | 13.15 | 11,488 | 28.23% | 15.14 | 12,895 |
| all_one | 28.23% | 13.09 | 11,433 | 28.03% | 15.14 | 12,887 |
| singletons | 27.73% | 13.25 | 11,586 | 27.63% | 15.06 | 12,845 |
| candidate_counts | 28.03% | 12.87 | 11,350 | 28.03% | 16.00 | 13,221 |
| candidate_confidence | 28.53% | 13.20 | 11,541 | 28.33% | 15.38 | 13,021 |
| deepsets_confidence | 28.03% | 13.55 | 11,877 | 27.93% | 15.22 | 12,930 |
| deepsets_embedding | 28.53% | 13.38 | 11,734 | 28.43% | 14.93 | 12,802 |
| same_head_single | 28.03% | 13.32 | 11,605 | 27.93% | 15.20 | 12,918 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | +0.20 | [-0.40, +0.80] | 0.983 | 1.0000 |
| random | +0.20 | [+0.00, +0.50] | 1.001 | 1.0000 |
| candidate_confidence | -0.60 | [-1.50, +0.30] | 0.986 | 0.8588 |
| deepsets_embedding | -0.60 | [-1.40, +0.20] | 0.970 | 0.8405 |

### Qwen2.5-3B-Instruct, MATH → Omni-MATH

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 23.32% | 1.00 | 833 | | | |
| sc16 | 28.03% | 16.00 | 13,221 | | | |
| asc | 28.03% | 12.78 | 11,174 | | | |
| esc | 28.03% | 13.39 | 11,686 | | | |
| reasc_offline | 27.63% | 11.25 | 9,743 | | | |
| reasc_online | 27.63% | 11.08 | 9,601 | | | |
| asc_dev_tuned | 28.03% | 11.64 | 10,272 | | | |
| reasc_dev_tuned | 27.63% | 11.25 | 9,743 | | | |
| mixture | 28.33% | 13.46 | 11,785 | 28.23% | 14.55 | 12,588 |
| uniform | 28.33% | 13.53 | 11,851 | 28.23% | 14.54 | 12,584 |
| random | 28.23% | 13.47 | 11,794 | 28.13% | 14.56 | 12,593 |
| alpha1 | 28.03% | 13.21 | 11,570 | 27.93% | 14.57 | 12,576 |
| single_partition | 27.63% | 14.25 | 12,415 | 27.63% | 14.48 | 12,550 |
| all_one | 28.13% | 14.22 | 12,377 | 28.13% | 14.47 | 12,535 |
| singletons | 28.03% | 13.21 | 11,570 | 27.93% | 14.57 | 12,576 |
| candidate_counts | 27.93% | 10.61 | 9,498 | 28.03% | 13.90 | 12,076 |
| candidate_confidence | 28.33% | 13.56 | 11,881 | 28.13% | 14.74 | 12,738 |
| deepsets_confidence | 27.43% | 11.62 | 10,309 | 27.13% | 14.68 | 12,693 |
| deepsets_embedding | 26.93% | 13.39 | 11,747 | 26.93% | 14.59 | 12,650 |
| same_head_single | 27.53% | 14.13 | 12,343 | 27.43% | 14.74 | 12,718 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | +0.30 | [+0.00, +0.70] | 1.019 | 0.7500 |
| random | +0.10 | [-0.20, +0.50] | 0.999 | 1.0000 |
| candidate_confidence | +0.00 | [-0.80, +0.80] | 0.992 | 1.0000 |
| deepsets_embedding | +1.40 | [+0.30, +2.60] | 1.003 | 0.0972 |

### Qwen2.5-7B-Instruct, MATH500

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 74.60% | 1.00 | 567 | | | |
| sc16 | 79.80% | 16.00 | 9,111 | | | |
| asc | 79.80% | 7.30 | 4,894 | | | |
| esc | 79.80% | 7.70 | 5,176 | | | |
| reasc_offline | 78.00% | 4.46 | 3,112 | | | |
| reasc_online | 79.60% | 5.73 | 3,993 | | | |
| asc_dev_tuned | 79.60% | 3.90 | 2,789 | | | |
| reasc_dev_tuned | 78.00% | 3.17 | 2,285 | | | |
| mixture | 79.20% | 4.06 | 2,949 | 80.60% | 7.77 | 5,382 |
| uniform | 79.80% | 4.21 | 3,056 | 80.40% | 7.88 | 5,450 |
| random | 79.00% | 4.06 | 2,953 | 80.40% | 7.76 | 5,370 |
| alpha1 | 79.80% | 4.43 | 3,200 | 80.60% | 8.28 | 5,630 |
| single_partition | 79.80% | 4.30 | 3,113 | 80.80% | 7.79 | 5,387 |
| all_one | 79.00% | 4.23 | 3,076 | 80.40% | 7.79 | 5,388 |
| singletons | 79.80% | 4.43 | 3,200 | 80.60% | 8.28 | 5,630 |
| candidate_counts | 77.60% | 3.66 | 2,684 | 79.80% | 8.27 | 5,513 |
| candidate_confidence | 79.20% | 4.06 | 2,964 | 81.40% | 8.49 | 5,853 |
| deepsets_confidence | 79.40% | 4.30 | 3,109 | 80.00% | 9.13 | 6,054 |
| deepsets_embedding | 79.20% | 4.59 | 3,354 | 80.20% | 8.88 | 6,061 |
| same_head_single | 79.80% | 4.59 | 3,339 | 80.60% | 8.58 | 5,832 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.60 | [-1.60, +0.40] | 0.922 | 1.0000 |
| random | +0.20 | [+0.00, +0.60] | 0.999 | 1.0000 |
| candidate_confidence | +0.00 | [-1.00, +1.20] | 0.995 | 1.0000 |
| deepsets_embedding | +0.00 | [-1.00, +1.00] | 0.879 | 1.0000 |

### Qwen2.5-7B-Instruct, GSM8K

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 92.12% | 1.00 | 289 | | | |
| sc16 | 94.09% | 16.00 | 4,635 | | | |
| asc | 94.09% | 4.91 | 1,494 | | | |
| esc | 94.09% | 5.09 | 1,559 | | | |
| reasc_offline | 93.86% | 2.99 | 983 | | | |
| reasc_online | 94.16% | 4.10 | 1,294 | | | |
| asc_dev_tuned | 93.78% | 2.33 | 707 | | | |
| reasc_dev_tuned | 93.63% | 1.71 | 541 | | | |
| mixture | 93.56% | 2.18 | 650 | 93.78% | 2.85 | 911 |
| uniform | 93.56% | 2.22 | 668 | 93.71% | 2.93 | 940 |
| random | 93.48% | 2.18 | 651 | 93.78% | 2.83 | 904 |
| alpha1 | 93.56% | 2.27 | 692 | 93.48% | 2.94 | 952 |
| single_partition | 93.40% | 2.22 | 669 | 93.71% | 2.84 | 905 |
| all_one | 93.40% | 2.20 | 660 | 93.63% | 2.85 | 911 |
| singletons | 93.56% | 2.27 | 692 | 93.48% | 2.94 | 952 |
| candidate_counts | 93.48% | 2.13 | 631 | 93.63% | 2.59 | 804 |
| candidate_confidence | 93.40% | 2.16 | 646 | 93.71% | 2.86 | 914 |
| deepsets_confidence | 93.63% | 2.17 | 648 | 93.71% | 2.74 | 859 |
| deepsets_embedding | 93.56% | 2.30 | 703 | 93.40% | 2.95 | 955 |
| same_head_single | 93.48% | 2.20 | 662 | 93.71% | 2.88 | 922 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | +0.00 | [-0.38, +0.38] | 0.940 | 1.0000 |
| random | +0.08 | [+0.00, +0.23] | 0.998 | 1.0000 |
| candidate_confidence | +0.15 | [-0.15, +0.45] | 1.007 | 1.0000 |
| deepsets_embedding | +0.00 | [-0.38, +0.38] | 0.925 | 1.0000 |

### Qwen2.5-7B-Instruct, Omni-MATH

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 27.73% | 1.00 | 824 | | | |
| sc16 | 33.53% | 16.00 | 13,156 | | | |
| asc | 33.53% | 11.87 | 10,451 | | | |
| esc | 33.53% | 12.43 | 10,923 | | | |
| reasc_offline | 32.53% | 9.55 | 8,335 | | | |
| reasc_online | 32.93% | 10.17 | 8,944 | | | |
| asc_dev_tuned | 33.33% | 7.34 | 6,618 | | | |
| reasc_dev_tuned | 30.13% | 5.08 | 4,385 | | | |
| mixture | 34.13% | 12.27 | 10,715 | 34.13% | 15.13 | 12,850 |
| uniform | 33.73% | 12.25 | 10,675 | 33.73% | 15.19 | 12,870 |
| random | 34.53% | 12.24 | 10,670 | 34.43% | 15.13 | 12,849 |
| alpha1 | 34.03% | 12.68 | 11,095 | 34.33% | 15.30 | 12,874 |
| single_partition | 33.63% | 12.60 | 10,967 | 33.53% | 15.02 | 12,780 |
| all_one | 34.23% | 12.48 | 10,879 | 34.13% | 15.03 | 12,776 |
| singletons | 34.03% | 12.68 | 11,095 | 34.33% | 15.30 | 12,874 |
| candidate_counts | 33.53% | 11.73 | 10,419 | 33.53% | 16.00 | 13,156 |
| candidate_confidence | 33.23% | 12.65 | 11,096 | 33.33% | 15.50 | 13,018 |
| deepsets_confidence | 33.93% | 12.70 | 11,107 | 33.93% | 15.30 | 12,932 |
| deepsets_embedding | 33.33% | 13.12 | 11,453 | 33.43% | 14.83 | 12,719 |
| same_head_single | 33.73% | 13.15 | 11,484 | 33.73% | 15.29 | 12,908 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | +0.10 | [-0.80, +0.90] | 0.966 | 1.0000 |
| random | -0.40 | [-0.80, -0.10] | 1.004 | 0.5000 |
| candidate_confidence | +0.90 | [-0.10, +1.90] | 0.966 | 0.5000 |
| deepsets_embedding | +0.80 | [-0.50, +2.10] | 0.936 | 0.5824 |

### Qwen2.5-7B-Instruct, MATH → Omni-MATH

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 27.73% | 1.00 | 824 | | | |
| sc16 | 33.53% | 16.00 | 13,156 | | | |
| asc | 33.53% | 11.87 | 10,451 | | | |
| esc | 33.53% | 12.43 | 10,923 | | | |
| reasc_offline | 32.73% | 10.40 | 9,114 | | | |
| reasc_online | 32.93% | 10.17 | 8,944 | | | |
| asc_dev_tuned | 33.33% | 7.34 | 6,618 | | | |
| reasc_dev_tuned | 32.33% | 7.66 | 6,826 | | | |
| mixture | 33.03% | 8.48 | 7,390 | 33.73% | 13.59 | 11,795 |
| uniform | 33.03% | 8.69 | 7,592 | 33.53% | 13.71 | 11,924 |
| random | 32.93% | 8.42 | 7,324 | 33.73% | 13.59 | 11,802 |
| alpha1 | 33.53% | 9.41 | 8,255 | 34.03% | 13.63 | 11,815 |
| single_partition | 32.63% | 8.90 | 7,766 | 33.13% | 13.51 | 11,716 |
| all_one | 33.03% | 8.86 | 7,721 | 33.63% | 13.45 | 11,644 |
| singletons | 33.53% | 9.41 | 8,255 | 34.03% | 13.63 | 11,815 |
| candidate_counts | 33.23% | 6.77 | 6,211 | 33.53% | 13.14 | 11,471 |
| candidate_confidence | 33.03% | 8.36 | 7,400 | 33.63% | 14.02 | 12,231 |
| deepsets_confidence | 32.73% | 8.65 | 7,488 | 33.13% | 13.41 | 11,573 |
| deepsets_embedding | 33.43% | 9.98 | 8,703 | 33.33% | 14.14 | 12,319 |
| same_head_single | 32.63% | 9.70 | 8,465 | 32.83% | 13.96 | 12,119 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.50 | [-1.20, +0.10] | 0.895 | 0.9062 |
| random | +0.10 | [-0.20, +0.40] | 1.009 | 1.0000 |
| candidate_confidence | +0.00 | [-0.70, +0.70] | 0.999 | 1.0000 |
| deepsets_embedding | -0.40 | [-1.30, +0.50] | 0.849 | 1.0000 |

### Llama-3.1-8B-Instruct, MATH500

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 50.00% | 1.00 | 479 | | | |
| sc16 | 58.80% | 16.00 | 7,709 | | | |
| asc | 59.00% | 10.84 | 6,006 | | | |
| esc | 59.00% | 11.51 | 6,306 | | | |
| reasc_offline | 58.40% | 7.65 | 4,311 | | | |
| reasc_online | 58.20% | 8.73 | 4,843 | | | |
| asc_dev_tuned | 59.60% | 6.77 | 3,997 | | | |
| reasc_dev_tuned | 58.60% | 6.73 | 3,834 | | | |
| mixture | 60.80% | 6.67 | 4,062 | 60.60% | 10.29 | 5,963 |
| uniform | 60.40% | 6.28 | 3,839 | 60.40% | 10.29 | 5,963 |
| random | 60.80% | 6.64 | 4,047 | 60.40% | 10.28 | 5,959 |
| alpha1 | 59.20% | 11.92 | 6,479 | 59.20% | 10.08 | 5,802 |
| single_partition | 59.20% | 16.00 | 7,709 | 59.40% | 10.19 | 5,875 |
| all_one | 59.80% | 6.32 | 3,866 | 59.80% | 9.90 | 5,736 |
| singletons | 59.20% | 11.92 | 6,479 | 59.20% | 10.08 | 5,802 |
| candidate_counts | 59.40% | 6.44 | 3,892 | 59.20% | 11.02 | 6,118 |
| candidate_confidence | 60.80% | 6.75 | 4,057 | 60.20% | 10.65 | 6,170 |
| deepsets_confidence | 60.80% | 5.62 | 3,454 | 59.60% | 10.54 | 6,007 |
| deepsets_embedding | 59.20% | 10.46 | 6,014 | 59.20% | 10.46 | 6,014 |
| same_head_single | 59.40% | 6.73 | 4,173 | 59.20% | 11.25 | 6,363 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | +1.60 | [-0.40, +3.60] | 0.627 | 0.6746 |
| random | +0.00 | [+0.00, +0.00] | 1.004 | 1.0000 |
| candidate_confidence | +0.00 | [-1.40, +1.40] | 1.001 | 1.0000 |
| deepsets_embedding | +1.60 | [-0.40, +3.60] | 0.675 | 0.6746 |

### Llama-3.1-8B-Instruct, GSM8K

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 82.41% | 1.00 | 183 | | | |
| sc16 | 90.37% | 16.00 | 2,885 | | | |
| asc | 90.37% | 6.14 | 1,231 | | | |
| esc | 90.30% | 6.51 | 1,312 | | | |
| reasc_offline | 88.93% | 3.42 | 751 | | | |
| reasc_online | 89.84% | 4.50 | 952 | | | |
| asc_dev_tuned | 89.54% | 2.97 | 609 | | | |
| reasc_dev_tuned | 88.93% | 2.88 | 632 | | | |
| mixture | 89.39% | 2.92 | 600 | 90.14% | 3.71 | 785 |
| uniform | 88.78% | 2.65 | 537 | 90.07% | 3.83 | 815 |
| random | 89.46% | 2.92 | 601 | 90.22% | 3.72 | 788 |
| alpha1 | 88.93% | 2.67 | 537 | 89.92% | 3.66 | 772 |
| single_partition | 89.54% | 3.10 | 642 | 89.99% | 3.91 | 828 |
| all_one | 89.39% | 3.03 | 622 | 89.99% | 3.80 | 803 |
| singletons | 88.93% | 2.67 | 537 | 89.92% | 3.66 | 772 |
| candidate_counts | 89.46% | 2.95 | 607 | 89.61% | 3.45 | 729 |
| candidate_confidence | 88.40% | 2.66 | 538 | 89.76% | 3.59 | 758 |
| deepsets_confidence | 89.61% | 3.30 | 677 | 89.69% | 3.63 | 751 |
| deepsets_embedding | 89.08% | 2.68 | 540 | 89.84% | 3.83 | 809 |
| same_head_single | 89.61% | 3.12 | 644 | 89.99% | 3.89 | 823 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | +0.45 | [+0.08, +0.91] | 1.118 | 0.2109 |
| random | -0.08 | [-0.38, +0.15] | 0.999 | 1.0000 |
| candidate_confidence | +0.99 | [+0.45, +1.59] | 1.116 | 0.0039 |
| deepsets_embedding | +0.30 | [-0.15, +0.76] | 1.111 | 0.6875 |

### Llama-3.1-8B-Instruct, Omni-MATH

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 13.71% | 1.00 | 1,066 | | | |
| sc16 | 17.22% | 16.00 | 16,941 | | | |
| asc | 17.22% | 14.31 | 15,932 | | | |
| esc | 17.22% | 14.59 | 16,188 | | | |
| reasc_offline | 17.92% | 12.75 | 14,409 | | | |
| reasc_online | 16.92% | 13.02 | 14,697 | | | |
| asc_dev_tuned | 17.02% | 10.68 | 12,171 | | | |
| reasc_dev_tuned | 16.12% | 8.40 | 9,274 | | | |
| mixture | 17.42% | 13.97 | 15,988 | 17.62% | 15.37 | 16,820 |
| uniform | 17.42% | 14.09 | 16,083 | 17.52% | 15.41 | 16,830 |
| random | 17.42% | 14.01 | 16,014 | 17.62% | 15.35 | 16,816 |
| alpha1 | 16.92% | 14.17 | 16,165 | 17.12% | 15.40 | 16,804 |
| single_partition | 17.42% | 14.20 | 16,193 | 17.62% | 15.36 | 16,801 |
| all_one | 16.82% | 14.13 | 16,114 | 17.12% | 15.36 | 16,803 |
| singletons | 16.92% | 14.17 | 16,165 | 17.12% | 15.40 | 16,804 |
| candidate_counts | 17.42% | 14.07 | 15,771 | 17.22% | 16.00 | 16,941 |
| candidate_confidence | 17.42% | 14.29 | 16,256 | 17.42% | 15.49 | 16,854 |
| deepsets_confidence | 16.72% | 14.21 | 16,190 | 17.02% | 15.38 | 16,798 |
| deepsets_embedding | 17.62% | 14.36 | 16,293 | 17.62% | 15.29 | 16,782 |
| same_head_single | 17.62% | 14.50 | 16,418 | 17.82% | 15.69 | 16,887 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | +0.50 | [-0.20, +1.30] | 0.989 | 1.0000 |
| random | +0.00 | [-0.40, +0.40] | 0.998 | 1.0000 |
| candidate_confidence | +0.00 | [-1.10, +1.10] | 0.984 | 1.0000 |
| deepsets_embedding | -0.20 | [-1.40, +1.00] | 0.981 | 1.0000 |

### Llama-3.1-8B-Instruct, MATH → Omni-MATH

| Method | Selected: accuracy | responses | output tokens | γ=0.95: accuracy | responses | output tokens |
|---|---:|---:|---:|---:|---:|---:|
| pass1 | 13.71% | 1.00 | 1,066 | | | |
| sc16 | 17.22% | 16.00 | 16,941 | | | |
| asc | 17.22% | 14.31 | 15,932 | | | |
| esc | 17.22% | 14.59 | 16,188 | | | |
| reasc_offline | 17.12% | 13.30 | 14,986 | | | |
| reasc_online | 16.92% | 13.02 | 14,697 | | | |
| asc_dev_tuned | 17.02% | 10.68 | 12,171 | | | |
| reasc_dev_tuned | 17.22% | 12.37 | 13,994 | | | |
| mixture | 17.12% | 12.36 | 14,200 | 17.32% | 14.64 | 16,477 |
| uniform | 16.92% | 11.69 | 13,460 | 17.12% | 14.61 | 16,427 |
| random | 17.02% | 12.37 | 14,212 | 17.12% | 14.65 | 16,485 |
| alpha1 | 17.72% | 15.01 | 16,570 | 17.62% | 14.37 | 16,138 |
| single_partition | 17.52% | 16.00 | 16,941 | 17.32% | 14.35 | 16,140 |
| all_one | 17.12% | 11.60 | 13,321 | 17.32% | 14.31 | 16,130 |
| singletons | 17.72% | 15.01 | 16,570 | 17.62% | 14.37 | 16,138 |
| candidate_counts | 17.02% | 10.70 | 12,271 | 17.42% | 14.34 | 15,989 |
| candidate_confidence | 16.82% | 10.97 | 12,453 | 17.32% | 14.53 | 16,298 |
| deepsets_confidence | 17.02% | 10.96 | 12,765 | 17.32% | 14.53 | 16,342 |
| deepsets_embedding | 17.82% | 14.39 | 16,212 | 17.82% | 14.39 | 16,212 |
| same_head_single | 17.12% | 11.65 | 13,434 | 17.42% | 14.83 | 16,533 |

Mixture (selected) minus control (selected), paired by question:

| Control | Accuracy Δ (pp) | 95% CI (pp) | Output-token ratio | Holm p |
|---|---:|---|---:|---:|
| alpha1 | -0.60 | [-1.50, +0.30] | 0.857 | 0.9181 |
| random | +0.10 | [+0.00, +0.30] | 0.999 | 1.0000 |
| candidate_confidence | +0.30 | [-0.60, +1.20] | 1.140 | 1.0000 |
| deepsets_embedding | -0.70 | [-1.70, +0.30] | 0.876 | 0.9181 |

Figures: [figures/](figures/). Full metrics, seed stability, fixed-prefix metrics and threshold curves: [evaluations/](evaluations/). Per-question decisions are in the Hugging Face dataset (`decisions/`).
