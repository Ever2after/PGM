# ReASC-code setting에서 PGM mixture 재구현·비교 계획

작성: 2026-11-02 KST. 상태: **source audit·구현 및 실험 계획 작성 완료. Benchmark 구현·환경 설치·새 response 생성·학습·평가는 아직 시작하지 않았다.**

## 1. 목표와 재현 기준

질문: **공식 ReASC 코드의 동일 generation·answer extraction·test bank에서, 프로젝트의 best retained method인 positive-support PGM mixture가 accuracy–cost trade-off를 개선하는가?**

여기서 best retained는 프로젝트 내부 reference를 뜻한다. 아직 외부 benchmark SOTA를 입증한 결과가 아니다. 실패했던 V5 temporal-state head는 primary method에 포함하지 않는다.

기준 source:

- 공식 repo: `junseokkim00/ReASC`, commit `8ee1c76ea0c04bb3cdb81d09bc2655dbc3a8cecb`.
- Archive: `pgm_reasc.zip`, SHA256 `b4427aa382baf221df0a9d241c025241d46de121fee74d6cacc3181987c0ccab`.
- 논문: [Findings ACL2026 PDF](https://aclanthology.org/2026.findings-acl.1085.pdf).
- 구현의 차이와 근거: [reference audit](./REASC_REFERENCE_AUDIT.md).

**Primary baseline은 공식 code-faithful ReASC이다.** 논문 설명과 코드가 다른 confidence/window/normalization 동작을 임의로 고쳐서 원본이라고 부르지 않는다. `reference_code` 결과와 selected-answer·실제 비용을 명확히 한 `corrected_evaluation` 결과를 함께 기록한다. 후자는 공통 answer/stop decision을 유지하고 평가 방식을 구분하는 profile이며, ReASC의 confidence 변경은 별도 algorithm arm이다.

## 2. 첫 실행 범위: Gemma-3-4B, MATH

| 항목 | 고정할 설정 |
|---|---|
| Model | `google/gemma-3-4b-it`; 기존 cached revision `093f9f388b31de276ce2de164bdc2081324b9767`를 명시적으로 pin |
| Original reference revision | 공식 repo는 model revision을 pin하지 않음. 따라서 이 revision이 저자의 당시 checkpoint라고 주장하지 않음 |
| Runtime target | 별도 `.venv-reasc` 또는 isolated environment: vLLM0.11.0, Torch2.8.0, Transformers4.56.2, NumPy2.2.6, SciPy1.15.3, sklearn1.7.2 |
| Sampling | resolved model defaults를 dump하여 고정. Gemma 기준 temperature1.0, top-p.95, sampling top-k64를 확인 후 사용 |
| Requested logprobs |20. Sampling top-k64와 다른 설정이며 actual dictionary size도 저장 |
| Budget | 질문당 bank16; adaptive maximum16. ReASC 자체의 t1 stop 허용 |
| Output length | MATH1,024 tokens; 기존4,096-token bank 재사용 안 함 |
| Prompt | official user question+chat template. 추가 step-by-step/boxed suffix 안 붙임 |
| Seed | generation0, head training42. 모든 RNG state·request sequence 기록 |
| Parallelism | GPU4·5 visible, TP2를 우선 사용. 실행 시 실제 GPU availability와 memory 확인 후 고정 |
| Answer extraction | official 후속 user prompt, max128 tokens, resolved default sampling. 참조 pipeline의 inline pass와 후속 overwrite pass를 모두 추적 |
| Judge | official `xVerify-3B-I` interface. 실제 checkpoint·tokenizer·revision·decoding을 확보하여 pin; placeholder 그대로 실행하지 않음 |
| Serving capacity | context cap·KV cache·batching 값은 reference의 resolved 값을 우선 기록. 운영상 override가 필요하면 prompt+output capacity를 검증하고 차이를 명시 |

vLLM0.11.0이 현재 장비에서 실행되지 않으면, 임의로0.18을 동일 재현이라고 부르지 않는다. 변경 runtime에서 모든 방법에 동일 bank를 제공하는 **controlled port**로 별도 표기하고, 작은 deterministic/sampling smoke에서 차이를 기록한 뒤 진행한다. 현재 환경을 덮어쓰지 않는다.

## 3. Source compatibility에서 반드시 보존할 것

1. Native SC/ASC/ESC/ReASC를 동일 bank와 동일 parsed-answer class에 적용한다. 현재 우리 baseline의 minimum2/cap8을 재사용하지 않는다.
2. ReASC token score는 공식 top-logprob dictionary proxy, suffix 포함 window128, bottom `max(1,T//10)`를 그대로 사용한다.
3. Offline ReASC의 lowest-group calibration / Bottom10 inference mismatch와 fallback을 reference profile에 보존한다. 별도 Bottom10-consistent sensitivity에서만 바꾼다.
4. Online ReASC는 test question별 첫 response만으로 GMM을 fit한다. Labels와 response2..16 confidence를 읽지 않는다. 이는 cohort 전체의 첫 pass가 필요한 transductive calibration이며 strict single-query streaming과 구분한다.
5. 공식 ASC의 Beta probability는 `1-betainc(a+1,b+1,.5)`로 옮기되 normal-range에서 원래 numerical integral과 parity를 검사한다. Overflow·zero-std·empty response는 임의 drop 없이 별도 flag와 수학적으로 동등한 안정화로 처리한다.
6. ESC는 disjoint4-window가 stop trigger이며, stop 시점까지 누적 prefix majority를 answer로 선택한다.
7. Original prediction grouping은 reference `clean(response)` → `clean(pred)` → raw pred fallback과 first-observed tie를 사용한다. PGM도 동일 grouping을 사용한다. PGM만 symbolic-equivalence로 answer classes를 더 합치지 않는다.
8. Archive generator의 argmax-fallback logprob은 PGM input으로 사용하지 않는다. 실제 `seq.token_ids[t]`의 logprob을 저장하고 exact-ID teacher forcing으로 검증한다. 이 수정은 ReASC reference confidence proxy를 바꾸지 않는다.
9. Gold는 별도 sidecar에 저장하고 generator·feature encoder·policy inference interface에 전달하지 않는다. Judge/verdict는 supervision·평가에서만 읽는다.

## 4. 데이터 설계와 규모

먼저 official calibration128 IDs를 고정한다. 원래 code의 seed0·sample ordering과 initialization에 따른 RNG state를 기록하며, 독립 RNG를 사용하는 경우 그 변경을 manifest에 명시한다. 그다음 train/dev/fresh-test를 서로 겹치지 않게 선정한다.

| Role | Source | Questions | Responses/q | 새 responses | Label use |
|---|---|---:|---:|---:|---|
| PGM train | 공식 development pool 중 original-MATH-train ID, 과거 cohort와 calibration 제외 |800|16|12,800|Support/gate/OTHER/alpha 및 continuation fit |
| PGM dev | 같은 original-train pool, train과 분리 |200|16|3,200|Checkpoint·policy selection |
| Calibration | 공식12,000-row development pool에서128, benchmark/test/train/dev와 분리 |128|16|2,048|ReASC-offline은 first response128개만 사용; PGM T는 prefix1..16 사용 |
| MATH500 benchmark | official `math500/test.jsonl`의500문제 전부 |500|16|8,000|Frozen-policy benchmark evaluation |
| Fresh confirmatory holdout | original-MATH-test ID 중 registry·calibration·MATH500 제외 |500|16|8,000|Locked confirmatory evaluation |
| 합계 | |**2,128**|16|**34,048**|Extraction/judge 호출은 별도 |

- 공식 development pool은 train/test가 섞인12,000행이므로 전체를 original-MATH-train이라고 부르지 않는다. Normalized unique11,998개다.
- 현재 historical registry를 제외한 original-train unique6,174개와 original-test unique1,150개가 남아 있다. Calibration을 먼저 제외해도 위 규모를 구성할 수 있다. 선택 시 official `unique_id` prefix와 pinned original HF source를 problem hash로 교차 검증한다.
- MATH500은 이전 연구 과정에서 이미 평가했다. 새 generation이라도 처음 보는 blind dataset이라고 부르지 않는다. 이를 보완하는 fresh500은 기존 모든 cohort와 normalized overlap0을 요구한다.
- Smoke16문제는 dev에서 미리 지정한 subset을 사용해 추가 test selection을 하지 않는다. 실제 matched-setting token bank가 재사용 가능하면 본 실험 bank에 포함하고, 별도 settings이면 smoke-only로 보관한다.
- Calibration bank의 first response128개는 공식 calibration generator처럼 서로 다른 question을 batch16으로 처리해 먼저 생성한다. PGM용 additional15 responses는 그 뒤에 생성하여 first-response cache를 바꾸지 않는다. 모든 RNG·request ordering 차이를 manifest에 남긴다.
- ReASC-offline의 labeled128-response calibration과 PGM의 train/dev/prefix supervision은 규모가 다르다. **같은 test setting과 동일 label budget은 다른 조건**이다. Training·dev·calibration question/response label 수와 head fitting cost를 결과 표에 반드시 병기한다.
- 이 단계에서 calibration128 IDs, train/dev IDs, fresh holdout IDs를 실제로 선정하거나 gold에 기반한 filtering을 하지 않는다. 구현 preflight에서 source·ordering·dedup 규칙과 함께 고정한다.

## 5. PGM mixture 재구현

기존 best reference의 active structure를 유지한다.

- Positive support MLP9→32→1, softplus.
- Six partitions: all-one + train-only cosine threshold4개 + singletons; duplicate partition mask.
- Gate MLP12→32→1, OTHER MLP25→32→1, learned alpha.
- Train-only final-layer reasoning token-mean embedding, cosine thresholds와 feature scalers.
- Neural heads는 candidate/OTHER NLL로 jointly fit. 새로운 error verifier, PRM, V5 temporal head, prompt 변경을 primary에 추가하지 않는다.

$$
e_{m,t}(a)=\sum_{i\le t}\mathbf1[a_i=a]\,n_{m,z_i,t}^{\alpha-1}\operatorname{softplus}(f_\theta(x_i)),
\qquad
p_t(a)=\sum_m\pi_{m,t}\,p_{m,t}(a).
$$

구현상 바뀌어야 할 부분:

1. **Bank length16과 model capacity32를 분리한다.** 기존 state_dict·OTHER slot32를 유지할 수 있지만 question당 prefix row 수를32라고 가정하는 train batching·reshape를 제거해야 한다. Padding은 t 이후 값의 접근을 허용하지 않는다.
2. New bank는 official extraction으로 answer를 얻는다. Candidate ID는 그 pred grouping을 쓰되, primary nine confidence features는 original reasoning response의 token ID에서 계산한다. Body에 boxed answer가 없으면 기존 missing-answer feature를 유지한다. Follow-up extraction token을 원래 answer-span feature에 몰래 섞지 않는다.
3. Embedding과 entropy·logprob은 exact generated IDs에 대한 TF pass로 추출한다. Native dictionary confidence와 full-vocabulary entropy를 명확히 다른 field에 저장한다.
4. Loss는 t1, t2, t3–4, t5–8, t9–16의 **5개 active buckets**에 equal total mass를 준다. 기존17–32 bucket을 포함한 채 weight나 sample count를 잘못 계산하지 않는다.
5. Threshold quantile와 scaler는 새 train800에서만 fit한다. 기존 temperature.7 feature 통계·weights를 primary retrained method의 초기 결과로 혼용하지 않는다. Pair sampling 수는 실제16-response bank에서 산출해 기록한다.
6. AdamW lr.001/wd.001, seed42, max120epochs, patience20, question batch32, gradient clipping1을 기본으로 고정한다. Dev prefix-balanced NLL로 early stopping한다. Loss·optimizer 변경은 후속 ablation이다.
7. Global temperature는 calibration128의 all16 prefixes에서 fit한다. Per-question weighting을 유지해 prefixes를 독립 question처럼 세지 않는다.
8. Full method는 horizon4 ExtraTrees continuation을 포함한다. Train800에서 positive horizon인 t<cap만 fit한다. Cap16이면12,000 rows, cap8이면5,600 rows다. Leaves32/128 선택과 policy search는 dev200에서 수행한다.
9. Primary continuation은 기존 best method와 같이 train selector의 fitted traces에서 fit한다. 이를 OOF라고 표현하지 않는다. OOF-selector trace로 바꾸는 실험은 별도 sensitivity다.
10. PGM policy는 minimum2/patience1, cap∈{8,16}, gamma∈{.5,.6,.7,.8,.85,.9,.925,.95,.975,.99,.995,disabled}, continuation lambda∈{0,1e-6,3e-6,1e-5,3e-5,1e-4}. Dev에서 SC16 accuracy−1%p 이상 중 reference consumed-chat cost가 최소인 설정을 선택한다. Feasible 후보가 없으면 실패를 기록하며 기준을 사후 완화하지 않는다.

Dev200은 checkpoint와 policy 선택에 재사용된다. Dev 통계를 독립 confirmatory evidence로 해석하지 않는다. Fresh holdout은 모든 fit·selection이 끝난 뒤 평가한다.

## 6. 비교 방법과 변경하지 않을 조건

| Group | Method | 설정·역할 |
|---|---|---|
| Main | Pass@1 | 첫 response |
| Main | SC16 |16개 majority |
| Main | ASC-code | Native .95 Beta, cap16 |
| Main | ESC-code | Native disjoint4 stop + prefix-majority answer |
| Main | ReASC-offline-code | First-response cal128, target.90, lambda.7, Beta.95, cap16, t1 가능 |
| Main | ReASC-online-code | Test first-response GMM, target.90, lambda.7, Beta.95, cap16, t1 가능 |
| Main | **PGM-mixture-full** | 새 setting에서 retrain한 기존 mixture + continuation, dev에서만 선택 |
| Diagnostic | Frozen-v2 transfer | 기존 checkpoint·T·policy를 그대로 새 bank에 적용; retrained result와 구분 |
| Ablation | PGM-mixture-fixed95 | Same selector, T만 calibration, minimum2/cap16/threshold.95, continuation 없음 |
| Ablation | PGM-mixture-no-value | Same selector, dev-selected gamma, cap16, continuation 없음 |
| Ablation | PGM-fixed-single | Same support/OTHER/alpha, partition slot1 고정, calibration·policy opportunity 동일 |
| Ablation | PGM-matched-random | Same cluster-size distribution control, train/dev budget 동일 |

모든 방법에 동일 sample text·order·pred grouping·judge 결과를 제공한다. Baseline을 PGM의 minimum2/cap8로 제한하지 않는다. Same-head single arm에서 모든 partitions를 계산한 상태로 optimized single-partition serving 성능이라고 주장하지 않는다.

Sensitivity는 primary lock 뒤 사전 정의된 순서대로 수행한다: (a) offline Bottom10 calibration consistency, (b) full-vocabulary confidence, (c) 동일 Stage1 gate를 공유한 Stage2 comparison, (d) OOF continuation. 여러 변경을 한꺼번에 넣고 원래 ReASC 또는 원래 PGM이라고 부르지 않는다.

## 7. 평가·비용·통계

두 평가 profile을 모두 저장한다.

### Reference-code profile

- Native parser/grouping, prefix/whole-bank verdict majority와 True tie, consumed chat-template token count를 그대로 재현한다.
- `TF_code = rounded_parameter_count × consumed_chat_token_count / 10^12`로 표시한다. 이는 source notebook proxy이며 일반적인 end-to-end FLOPs가 아니다.
- Native ESC의 full-bank verdict 집계는 report compatibility에만 사용한다. 미래 response·verdict를 inference에 넘기지 않는다.

### Corrected evaluation profile

- 같은 selected answer와 stop time을 사용하되, 선택된 **candidate answer 자체를** 공통 xVerify 설정으로 판정한다. 동일 question/candidate를 cache해 judge noise를 공유하고 deterministic decoding을 고정한다. 이것이 원본 full-response judge metric과 다른 점을 표시한다.
- Supervision에도 같은 canonical candidate-level label을 사용한다. Multiple correct strings가 있는 경우 기존 first-observed target·gold-group semantics를 명시하고 ambiguous cases를 보고한다. Label/version이 다른 결과를 하나의 headline으로 섞지 않는다.
- Primary scientific contrast는 fresh500의 PGM-mixture-full vs ReASC-offline-code다. ReASC-online과 SC16 등은 predefined secondary comparisons다. MATH500 결과는 별도 benchmark table이다.
- Accuracy, mean consumed responses, reasoning generated tokens, reference proxy, extraction input/output tokens, TF feature tokens/time, head/GMM overhead를 각각 보고한다.
- `2N FLOPs/token`은 별도 dense-model approximation으로 계산하며 reference proxy와 coefficient를 맞춘다. TF pass에는 재입력한 prompt/response token을 포함한다. Attention·KV·hardware effects를 실제 profiler 없이 정확한 FLOPs라고 부르지 않는다.
- PGM은 embedding·full-vocabulary feature TF pass를 추가로 요구한다. **Generated tokens만 줄고 feature 비용이 더 크면 end-to-end efficiency 개선으로 결론 내리지 않는다.**
- Original shell은 extraction을 두 번 실행하므로 both passes를 trace하고 consumed sample별 비용에 반영한다. 불필요한 overwrite pass를 제거하는 serving 최적화는 별도 experiment이며 reference generation RNG·pred 차이를 설명한다.
- Training·calibration 및 judge의 총 setup cost를 별도로 보고한다. Judge는 평가 전용이며 policy inference 비용과 구분한다. Train-free ReASC와 supervised PGM의 setup burden을 숨기지 않는다.
- Question-level paired bootstrap10,000회,95% CI를 사용한다. Accuracy 차이·cost ratio CI, paired discordances/McNemar를 보고한다. Prefix나 response를 독립 question으로 resample하지 않는다.
- Practical target: accuracy≥ReASC-offline−1%p 및 reference cost≤90% ReASC-offline. Statistical non-inferiority는 accuracy difference CI lower bound≥−1%p일 때만 선언한다. Point target 통과와 population guarantee를 구분한다.
- Seed0 primary를 고정한다. Training seeds43/44는 같은 bank의 robustness checks이며 best seed 선택을 하지 않는다. Generation seeds1/2가 추가되면 seed별 결과와 variability를 분리한다.

Bank replay는 실제 online latency가 아니다. Serving latency는 causally stop하는 runner에서 별도로 측정할 때만 보고한다. Online GMM variant는 cohort first-pass 비용과 calibration barrier도 포함한다.

## 8. 구현 순서·산출물

기존 source·locks를 바로 고치지 않고 신규 comparison entrypoint에서 adapter를 연결한다. 아래 구현은 2026-11-02에 추가했다. 실행 상태는 `artifacts/runs/reasc_pgm/progress.md`, 상세 환경·적용 차이는 `docs/REASC_PGM_EXECUTION.md`에서 추적한다. 전체 성능 결과는 아직 나오지 않았다.

| 순서 | 작업 | 예정 산출물·완료 기준 |
|---|---|---|
|0|Reference pin·static audit|이번에 완료: 두 source, commit/hash, differences, dataset availability |
|1|Runnable reference policy port|`policies/reasc_reference.py` (ESC 포함); notebook oracle와 answer·stop·proxy cost parity |
|2|Shared generation+extraction bank|`collection/reasc_bank.py`; model/default/RNG manifest, token IDs, sample IDs, top-dictionary, both extraction traces, resume/dedup |
|3|Judge·canonical labels|`collection/reasc_judge.py`; xVerify revision 확보, raw verdict와 candidate verdict 분리, no-gold inference test |
|4|PGM16-prefix adapter·feature bank|`data/reasc_bank.py`; capacity32/length16 분리, TF confidence/embedding alignment, prefix-causality tests |
|5|Retraining·calibration·policy freeze|`experiments/reasc_comparison.py`; train/dev/cal IDs, checkpoint·scaler·threshold·policy hash를 test outcome 전에 lock |
|6|Paired benchmark·fresh holdout replay|`evaluation/reasc_comparison.py`; main/ablation tables, paired CIs, failure cases, full cost breakdown |
|7|Robustness·확장|Training seed checks, GSM8K·Omni-Math·Qwen으로 확장. MATH500 test 결과로 primary hyperparameter를 다시 고르지 않음 |

Artifact 예정 경로는 `artifacts/data/reasc_pgm/`와 `artifacts/runs/reasc_pgm/`다. 기존 `artifacts/{data,runs}/pgm_asc`와 checkpoint·결과는 유지한다. 계획용 설정은 `configs/reasc_comparison.plan.json`이며 현재 CLI에 전달할 runnable config가 아니다.

## 9. Smoke와 실행 gate

학습·benchmark 전에 다음을 통과해야 한다.

- Official notebook의 순수 score/gate/stop 계산만 추출한 oracle와 합성 bank 비교: dictionary size20/21, short-suffix windows, floor rounding, offline min/Bottom10 mismatch, GMM fallback, first-response weighted-Beta stop, ESC earlier-majority vs unanimous-window case, tie handling.
- vLLM 실제 token IDs와 sampled logprob alignment. Top1 token이 아닌 sample을 반드시 포함해 argmax fallback 오류를 잡는다.
- Native top-dictionary logprob과 raw TF logprob의 분포 정의·온도처리 차이를 분리해 기록한다. 서로 다른 quantity를 무조건 같다고 assert하지 않는다.
- True token count1024 truncation, EOS, answer extraction failure를 고정하고 실패 question을 임의 제외하지 않는다. Empty output의 fallback을 preregister한다.
- PGM t 시점 input과 decision이 future response·gold·verdict를 바꿔도 불변. Fitting/calibration에서는 label 사용을 별도 API로만 허용한다.
- Cohort normalized overlap0, unique question/sample identity, no duplicate append. PGM train scaler/partition threshold가 dev/test를 읽지 않음.
-16-question smoke에서 generator·extractor·TF feature·judge throughput을 측정하고 response/token 기반 ETA와 storage budget을 산정한다. 지금 근거 없는 전체 ETA를 제시하지 않는다.
- xVerify model path/revision, model access, exact environment compatibility가 미해결이면 benchmark gate를 열지 않는다. Synthetic parity·adapter 작업은 독립적으로 진행할 수 있다.

## 10. 후속 확장

첫 비교가 끝나면 동일한 고정 protocol로 GSM8K, archive Omni-Math, Qwen2.5-3B/7B를 순서대로 확장한다. Omni-Math는 1,000 source rows/999 unique 문제를 구분하여 code-compatible999-group 결과와 row-preserving 결과를 각각 보고한다. GPQA는 tokenizer-dependent <2,000 prompt-token filter가 적용되므로 실제 ID·수량을 model별로 freeze한다. 27B 실행은 GPU memory와 xVerify resource를 확인한 뒤 별도 budget으로 잡는다.

이 문서는 결과를 예측하거나 실행을 완료했다고 선언하지 않는다. 현재 reference policy, shared-bank collector, judge, 16-prefix adapter, training/selection/lock, paired evaluation, persistent supervisor를 구현했고 smoke를 실행 중이다. Full comparison의 성공 기준과 데이터·cost 규칙은 test 전에 고정하며 실제 완료 여부는 실행 status와 결과 파일로 확인한다.
