# Experiment history

V2와 V5를 selector/state 단계로 통합한 뒤, 그 외 실험의 실행 파일·checkpoint·결과 bank는 삭제했다. 이 문서는 시도한 방법, 데이터 역할, 결과와 한계를 보존한다. 아래 원문에 나오는 이전 경로와 당시 진행 상태는 historical record이며 현재 실행 경로가 아니다. V1 representation/scaler/train/dev 및 V3 calibration 일부는 V2/V5의 필수 입력으로 이관했다.

| Attempt | Method / decision | Outcome |
|---|---|---|
| Pilot / stage2 | vLLM generation, teacher-forcing features, relation information | Backend와 feature feasibility 확인. 과거 입력 중 필요한 response만 공통 bank로 이관 |
| V1 | Positive-support partition mixture, calibration, online checks | V2 selector의 초기 model·representation을 제공 |
| V2 | Minimum/patience/cap, prefix-weighted NLL, continuation | 유지: 통합 selector stage. 원래 holdout500에서 accuracy82.4%, mean tokens3492.6 |
| V3 | Equation representation, likelihood/quality auxiliary, partition variants | Holdout500 PGM78.6%, random78.6%; grouping 추가 이득 입증 못함 |
| V4 | OOF quality, shared reliability, matched controls | Development gate 실패. Fresh response 수집·confirmatory test 없음 |
| V5 | Frozen selector + 3-state temporal PGM + fixed4/Bellman values | 유지: 통합 state stage. Fresh holdout1000에서 primary78.3% vs frozen selector81.0%; accuracy constraint 실패 |

V2·V3·V5 holdout은 서로 다른 cohort다. 서로 다른 cohort의 accuracy를 직접 순위로 해석하지 않는다. V5에서 이전 V2 holdout500은 dev로 전환했다.


---

## Historical source: `BASELINE_REPRODUCTION_NOTES.md`

# Adaptive 비교군 재현 기록

작성일: 2026-11-01 KST. 최종 benchmark 평가 전 구현·검증 기록이다.

## 원문과 confidence 정의

[ReASC v1](https://arxiv.org/html/2601.02970v1)의 식 (1)–(2), 부록 E.2를 확인했다. 이 방법이 사용하는 token confidence는 entropy나 생성된 token 하나의 log-probability와 다르다.

$$
c_i=-\frac1{|\mathcal V|}\sum_{v\in\mathcal V}\log p_i(v)
=\operatorname{logsumexp}(l_i)-\operatorname{mean}(l_i).
$$

128-token window의 stride-1 평균을 구한 뒤 낮은 10% 그룹을 평균한다. 구현상 그룹 수는 올림 후 최소 1개, 128토큰보다 짧은 응답은 전체를 한 그룹으로 정의했다. 특별 토큰을 제외한 생성 응답 전체를 사용한다. 이 세부 정의는 supplemental manifest에 저장한다. 원문은 confidence 표준화와 exponential evidence weight, 단일 응답 gate, Beta 종료를 사용하므로 해당 구성과 보정 절차까지 구현·검증해야 ReASC 비교가 완료된다.

[CGES v2](https://arxiv.org/html/2511.02603v2)의 부록 H에는 raw sampled-token probability 기반 geometric/arithmetic LNS가 명시돼 있다. 이 값은 이미 저장한 vLLM raw log-probability에서 계산할 수 있다. 후보 공간이 커질 때 과거 증거도 현재 후보 수로 다시 계산하는 동적 support 규칙을 구현해야 한다.

## 추가 추출 범위와 보존

개발 6,400응답과 calibration-A/B 각 4,800응답, 총 16,000응답을 대상으로 한다. 현재 PGM이 사용한 confidence와 임베딩, 원래 학습·선택·보정 lock은 변경하지 않는다. 기존 정확한 prompt/token ID에 대해 새로운 teacher-forced pass로 vocabulary 평균 log-probability를 구한다. raw sampled log-probability의 vLLM 차이도 응답별로 검증한다. 새 답변 생성이나 final test 접근은 없다.

산출물은 각 split의 `baseline_features/`에 response ID·source token hash·feature fingerprint와 함께 저장한다. GPU 4는 개발, GPU 5는 calibration-A/B를 처리한다. 이 추가 forward pass 비용은 비교군의 특징 계산 비용으로 별도 기록하며, PGM에 필요한 비용과 구분한다.

## 현재 완료 범위

- 본 PGM train/dev 선택 104개와 A/B 보정 완료.
- 고정 설정의 추가 시드 43/44 학습·보정 16개 완료.
- full-vocabulary confidence 수식과 window 계산의 단위 검증 2개 통과.
- 원문 기반 ASC/ESC/ReASC/CGES의 전체 재현·보정·비교와 최종 protocol lock은 아직 미완료.

## 비교군 구현 및 고정 완료

추가 confidence 16,000개와 비교군 구현·보정을 완료했다. ASC는 [원 논문](https://arxiv.org/abs/2305.11860)의 top-two Beta, ESC는 [원 논문의 Algorithm 1](https://proceedings.iclr.cc/paper_files/paper/2024/file/3fe2a777282299ecb4f9e7ebb531f0ab-Paper-Conference.pdf)에 따라 겹치지 않는 window와 만장일치 window의 답을 사용한다. 조기 종료를 비활성화한 ESC fallback은 SC@32다. ReASC는 원문의 lambda=0.7, gate 목표 정확도 0.9를 사용한다. CGES는 geometric/arithmetic LNS 중 dev NLL로 선택한다. 이진 count/confidence 모델은 같은 train의 질문과 confidence 특징으로 학습한다.

calibration 문항 수 150개와 공통 calibration-B threshold grid는 본 연구의 사전 계획에 따른 조정이다. 논문의 기존 결과 수치나 원 코드의 모든 설정을 그대로 복제했다고 주장하지 않는다. 각 방법의 규칙과 입력을 수식·인과적 prefix 단위 검증으로 확인했다. 재현 수식·정책 고정·기존 파이프라인을 포함한 프로젝트 테스트 39개를 통과했다.

최종 lock을 저장한 뒤 MATH-500 수집을 시작했다. 결과를 보고 비교군이나 PGM의 설정을 다시 선택하지 않는다. 아직 최종 평가 결과는 없다.

---

## Historical source: `PGM_ASC_FOLLOWUP_PLAN.md`

# PGM-ASC sequential follow-up experiments (v2)

Created 2026-11-01 KST, before new calibration/holdout response collection.

## Scope and separation

The inspected MATH-500 bank is development evidence only. Preserve all v1 models,
locks and primary results. Use the existing 800-question train and 200-question
dev banks for training. Select new policies on a fresh calibration cohort; open
the fresh holdout labels for evaluation only after all stage policies are locked.
This is a bounded seven-stage follow-up, not an unlimited parameter search.

Draw 150 calibration-A, 300 calibration-B and 500 holdout questions by seeded
uniform sampling without replacement from MATH test, excluding normalized
problem overlap with every previous local cohort, including MATH-500. Retain
Asymptote questions as text and report their subgroup separately. These cohorts
share the MATH benchmark source and may still overlap LLM pretraining; they are
independent of this project's earlier evaluated questions, not a new task domain.

Use the existing pinned model/data revisions, prompt, vLLM sampling, 32 responses,
token-ID-aligned teacher forcing, and supplemental baseline confidence. Gold
files are separate from generation inputs. Do not exclude failed extraction or
truncation cases from accuracy/cost. GPU 5 collects v2; GPU 4 resumes the original
online timing experiment under its unchanged policy.

## Sequential stages

| Stage | Only new component | Fitting/selection |
|---|---|---|
| S0 | Global temperature on fresh calibration-A | gamma on fresh calibration-B; min=1, patience=1, cap=32 |
| S1 | Minimum response count | min in {1,2,4,8}; recalibrate gamma |
| S2 | Confidence persistence | patience in {1,2,3}; retain S1 minimum; recalibrate gamma |
| S3 | Continuation budget cap | cap in {8,16,32}; retain accepted min/patience; recalibrate gamma |
| S4 | Prefix-bucket temperature and OTHER offset | fit on calibration-A, question-level OOF regularization selection; gamma on B |
| S5 | Equal-total-weight prefix-bucket NLL | refit the same architecture on old train, early stopping on old dev; calibration A/B as above |
| S6 | Learned continuation value | future-prefix targets from old train only; choose head complexity on old dev, lambda/gamma on B |

Every stage includes its preceding accepted policy as a candidate. Adopt a change
only when it meets the empirical accuracy constraint and reduces calibration-B
mean tokens, or repairs an infeasible previous policy. Otherwise preserve the
previous policy, record the rejection, and continue. This greedy sequence measures
local additions; it does not identify the globally optimal combination.

S0 is a domain-matched recalibration control. Also report the original v1 frozen
policy separately, so new calibration data and policy changes are distinguishable.
Global and bucket calibrators use identical fresh calibration-A questions.

## Shared controls and selection rule

Run stages S0–S4 for mixture, single partition, alpha=1, size-matched random
grouping, candidate MLP, CGES, ASC and ReASC. For distribution-producing models,
apply the same global/bucket calibration opportunity; ASC/ReASC retain their
paper-derived confidence scores and receive the same minimum/patience/cap/gamma
search. ReASC's confidence normalization and first-response gate are fit on A.
Refit S5 for mixture, single, alpha=1 and random with their previously selected
architecture/lr/weight decay (no new architecture grid). Other controls retain
their accepted S4 policy. S6 applies the same continuation learner to mixture,
single, alpha=1, random, candidate MLP, CGES and ASC. ReASC retains its previous
policy at S6: the old train bank lacks its full-vocabulary confidence features,
and collecting another 25,600 full-vocabulary passes is outside this run. It
remains a fully calibrated S0–S4 control, not a matched S6 learner control.

Select gamma from {0.50,0.60,0.70,0.80,0.85,0.90,0.925,0.95,0.975,0.99,0.995,
no early stop}. Among policies with calibration-B accuracy at least SC32 minus
0.01, choose smallest generated-token cost, then highest accuracy, then fewer
responses, then fixed candidate order. If none qualifies, retain the most accurate
candidate and explicitly mark it infeasible. These are empirical constraints,
not finite-sample or distribution-shift guarantees. Model feature-computation
cost is separate from generated-token cost.

For S4 use buckets {1}, {2}, {3–4}, {5–8}, {9–16}, {17–32}. Shrink temperature
towards a global temperature and OTHER bias towards zero. Select shrinkage from
{0.01,0.1,1} using 3-fold question-level OOF NLL on calibration-A, then refit on
all A. Preserve the global calibrator as a B-selected control. Never tune on test.

S5 weights each of the six prefix buckets equally (equal total mass, not each
prefix equally). Fit from the same seed-42 initialization using the v1 optimizer
settings, at most 120 epochs, patience 20, and bucket-weighted dev NLL. All other
architecture and representation inputs are held fixed. Do not mix future
responses into current-prefix inputs.

S6 targets corrected-minus-regressed correctness after four more responses,
clipped to the preceding stage's accepted cap, and the corresponding token
increment. Train/validate the head only on prefixes before that cap.
Use an ExtraTrees regressor with fixed
random seed, 100 trees, max_depth=8, and min_samples_leaf selected from {32,128}
by old-dev squared error. Input only current/past confidence, OTHER, answer
stability, prefix count and count/confidence summary. All methods receive the
same feature family; no gold or future tokens at inference. The shared summary
includes teacher-forcing confidence, so S6 controls also incur that feature cost.
Underlying answer selectors' train predictions are in-sample; the continuation
head's complexity is validated on separate dev questions. OOF selector targets
are a later control if this initial head is useful. Fit targets on train;
calibrate policy lambda from a fixed grid on B. Stop when estimated value is at
most lambda times estimated token cost, or when the ordinary stop/cap fires.
Retain the original policy as a control. Report extra head runtime separately
before making end-to-end latency claims. This is a finite-horizon proxy, not a
solution to the full optimal-stopping problem.

## Final evaluation and decisions

Freeze all stages, model/calibrator/head hashes and implementation hashes before
generating/evaluating the 500-question holdout. Evaluate every locked stage once;
do not promote/retrain using holdout results. Report paired question-bootstrap
accuracy/token differences versus the previous stage, SC32, and the strongest
adaptive control chosen on B; report confidence intervals and point-estimate
constraint failures. Report fixed budgets 8/16/32 and extraction/truncation rates.
Exploratory multiple contrasts are not confirmatory significance claims.

The first v2 run estimates the sequential effects. Extra training seeds, response
order sensitivity, semantic-pair annotation, grounded verifiers, and optimized
online serving are later experiments, triggered by the measured remaining failure
modes. Do not invent error-pattern labels from answer equality. Gate/representation
changes require evidence beyond generic stopping-policy improvements.

## Artifacts and execution

- Config: `configs/pgm_v2.json`
- Driver: `pgm_pilot/pgm_followup.py`
- Policies/calibration/training: `pgm_pilot/pgm_followup_methods.py`
- Run root: `artifacts/runs/gemma3_pgm_v2/`
- Status: `status.json`; stage decisions: `selection_lock.json`; result: `report.md`

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u -m pgm_pilot.pgm_followup --gpu 5
```

The driver is resumable and uses a process lock. A running phase has a live PID
and updated progress; a stale JSON status is not evidence that a job is running.

---

## Historical source: `PGM_ASC_V2_REDESIGN_OPTIONS.md`

# Frozen v2 이후의 PGM 재설계 후보

작성: 2026-11-02 KST. 기존 구현·평가 결과와 spent v2 holdout의 추가 diagnosis에 근거한 설계 검토다. 새 학습이나 generation은 실행하지 않았다. 아래 새 구조의 우수성·novelty는 입증된 것이 아니다.

## 1. 판단

**먼저 v2의 partition mixture를 제거한 single fixed partition을 기준으로 단순화하고, 다음 PGM의 중심을 partition identity에서 candidate coverage와 sequential decision state로 옮기는 방향을 우선 검토한다.**

비용 개선과 accuracy 개선은 구분한다. 비용은 confidence/futility와 future value의 구분이 핵심이고, 큰 accuracy 개선에는 selector가 사용할 추가 정보가 필요하다. Graph를 바꾸는 것만으로 correct answer가 새로 생성되거나 reasoning correctness가 판별되지는 않는다.

## 2. 확인한 근거와 개선 범위

### 2.1 기존 single control이 이미 있다

V2 final의 같은 500-question holdout:

| Method | Accuracy | Mean generated tokens |
|---|---:|---:|
| Mixture | 82.4% | 3,492.6 |
| Single fixed partition | 82.4% | 3,451.1 |
| CGES | 82.6% | 3,032.4 |

실제 single checkpoint는 `mode=single, partition=1`이다. All-one slot 0이 아니라 raw hidden-state cosine threshold 약 0.9656을 쓰는 첫 clustering rule이다. Rule 하나를 모든 question에 공통 적용하며, 실제 cluster assignment는 question과 prefix에 따라 바뀐다.

Mixture−single accuracy interval은 [-1.2,+1.2]%p, token difference interval은 [-94.5,+198.1]이다. 동등성이나 비용 우위를 입증한 것은 아니다. 더구나 mixture는 global temperature, single은 bucket calibration이고 continuation head도 각각 학습했다. 따라서 이 비교는 gate만 제거한 clean ablation이 아니다.

V4의 H4 single은 다른 quality/likelihood model이고 all-one을 선택했다. 그 결과로 v2 positive-support single을 대체 평가하지 않는다.

### 2.2 V2 error 88개

- 27개: stop 시 correct candidate가 이미 관측됐다. Candidate selection의 대상이다.
- 61개: stop 시 correct candidate가 아직 없었다. 동일 시점의 selector만 바꿔서는 고칠 수 없다.
- 그 61개 중 correct answer가 관측되는 수는 t=8까지 9개, t=16까지 17개, t=32까지 24개였다.
- 같은 selector가 t=32에서 correct이 되는 기존 error는 13개였다. Sampling 연장의 효과를 correct-candidate 출현과 동일시하면 안 된다.

Fixed cap=8에서 현재 selector를 유지하고 question별로 t=2~8 중 정답이 되는 stop time을 oracle처럼 골라도 accuracy upper bound는 **84.4%**다. 현재 82.4%에서 +2.0%p다. 미래를 알고 stop하는 oracle이므로 실제 달성 가능성이나 일반화 성능을 보장하지 않는다.

반면 현재 frozen stopping time을 유지하고 observed candidate 중 항상 correct을 고르는 oracle이면 87.8%다. 27개 selection error를 고친 수학적 ceiling이며, 이를 달성하는 정보가 현재 feature에 있다는 뜻은 아니다.

### 2.3 Confidence stop과 futility stop

| Actual stop reason | Questions | Wrong | Wrong: gold present / absent | Wrong 중 같은 selector로 t≤8의 나중 시점에 correct이 되는 수 |
|---|---:|---:|---:|---:|
| Confidence | 427 | 34 | 6 / 28 | 4 |
| Continuation value | 31 | 25 | 10 / 15 | 2 |
| Cap | 42 | 29 | 11 / 18 | 0 |

이유별 category는 서로 배타적이다. 실제 t=8에 도달한 question은 44개로, 위 cap category 외에 confidence exit 2개를 포함한다.

Continuation exit의 25/31이 wrong이라고 해서 25개 모두 부적절한 stop인 것은 아니다. 나머지 sampling의 expected benefit이 낮아서 종료한 futility decision일 수 있다. 실제로 같은 selector로 t≤8의 나중 시점에 correct이 되는 것은 2개였다.

현재 S6는 confidence threshold를 통과하거나, continuation predicted value≤lambda×predicted tokens이면 종료한다. 두 stop의 의미를 하나의 confidence로 해석하지 않아야 한다.

[추가 diagnosis JSON](./artifacts/analysis/frozen_v2_redesign/diagnostics.json) · [v2 보고서](./artifacts/runs/gemma3_pgm_v2/report.md)

## 3. 후보 A: Single fixed partition — 가장 먼저 할 단순화

현재:

$$
p(Y=y\mid D_t)=\sum_m\pi_m(D_t)p(Y=y\mid D_t,z^{(m)}).
$$

Global fixed rule:

$$
p(Y=y\mid D_t)=p(Y=y\mid D_t,z_{\eta^*}(D_t)).
$$

eta*는 train/dev로 한 번 정하고 fresh evaluation에서 바꾸지 않는다. Partition ID uncertainty를 제거해도 answer uncertainty와 cluster-level latent variable을 모델링하는 PGM은 구성할 수 있다. 여러 partition의 weight가 PGM의 필수 요건은 아니다.

구분해야 할 세 경우:

1. **Global fixed clustering rule:** 하나의 threshold를 모든 question에 사용한다. 가장 우선한다.
2. **Hard per-question/prefix selection:** argmax gate로 하나만 고른다. Gate와 여러 partition 계산이 여전히 필요하고 early-prefix selection이 불안정할 수 있다.
3. **All-one cluster:** 모든 response를 한 cluster로 묶는다. Clustering 자체를 제거하는 control이다.

All-one에서 기존 support score는 다음처럼 된다.

$$
S_t(y)=t^{\alpha-1}\sum_i s_i\mathbf1[a_i=y].
$$

Support s_i를 고정하면 observed-answer ranking은 independent weighted voting과 같다. OTHER score와 confidence는 달라질 수 있다. 이 결과를 reasoning-grouping의 효과로 주장하면 안 된다.

검증은 두 부분으로 나눈다. 먼저 support/OTHER/scaler를 고정하고 mixture→single만 바꿔 fixed-budget distribution과 winner 변화를 본다. 다음으로 모든 arm에 같은 calibration·continuation 학습 기회를 주고 실제 사용 정책을 비교한다. 옛 continuation head를 변경된 confidence input에 그대로 붙인 결과만으로 판단하지 않는다.

Single도 hidden-state partition을 사용하면 기존 TF extraction 비용이 남는다. Clustering/gate 계산 감소를 end-to-end speedup으로 간주하지 않고 별도 측정한다.

## 4. 후보 B: Coverage–selection PGM — answer distribution의 역할 분리

Latent variable을 다음처럼 둔다.

$$
E_t=\mathbf1[Y\in C_t],\qquad
u_t=p(E_t=1\mid D_t).
$$

Observed candidates 안의 conditional selector를 v_t(y)라 하면:

$$
p(Y=y\mid D_t)=u_t v_t(y),\quad y\in C_t,
\qquad p(Y=\mathrm{OTHER}\mid D_t)=1-u_t.
$$

이 구조는 모든 observed answer를 낮은 score로 줄이는 경우와 correct answer 자체가 없을 가능성을 명시적으로 분리한다. Fixed-partition v2나 다른 selector를 v_t로 사용할 수 있다.

중요한 한계: 이 factorization만으로 model capacity나 정보가 늘지는 않는다. Loss BCE(E_t,u_t)+E_t×CE(Y,v_t)는 joint answer NLL의 분해다. 개선 가설은 별도 coverage feature·supervision·temporal constraint에서 나온다.

Coverage에는 new-answer arrival, confidence distribution, valid/truncated fraction, leader stability 등을 입력한다. Train gold는 E_t target에만 사용한다. OTHER가 이미 v2에 있으므로 이것을 완전히 새로운 unseen-answer 개념으로 포장하지 않는다. 현재 결합된 logit head를 더 검증하기 쉬운 구조로 바꾸는 제안이다.

## 5. 후보 C: Three-state temporal PGM — 가장 우선하는 새 PGM

현재 선택한 candidate를 a_hat_t로 두고 state를 다음처럼 정의한다.

| State | 의미 |
|---|---|
| U | Correct candidate가 아직 관측되지 않음 |
| W | Correct candidate는 관측됐으나 현재 선택은 wrong |
| R | 현재 선택이 correct |

이 상태는 train에서만 gold로 라벨링하고, inference에서는 posterior belief를 계산한다. 이를 모든 question에서 관측 가능한 상태로 취급하지 않는다.

Conditional chain의 filtering family:

$$
b_t(s)\propto\psi_\theta(s,o_t)
\sum_{s'}T_\theta(s',s\mid o_t)b_{t-1}(s').
$$

o_t는 새 response의 confidence, 새 answer 출현, 현재 answer identity 변경 등 causal observed information이다. Prefix summary를 포함하면 독립 emission의 생성모형으로 해석하지 않고 conditional chain으로 학습한다.

논리적 transition mask:

- W/R→U는 금지한다. 한번 관측한 correct candidate는 observed set에서 사라지지 않는다.
- New answer가 없으면 U→W/R는 금지한다.
- 선택한 answer identity가 같으면 W↔R는 금지한다. 고정된 answer의 correctness가 바뀌지는 않는다.
- Belief b_t(U)는 새 observation을 조건으로 다시 계산되므로 항상 단조 감소하도록 강제하지 않는다. 실제 hidden coverage state의 monotonicity와 posterior probability의 monotonicity는 다르다.

이 구조는 graph의 상태를 현재 model의 실패 유형에 직접 대응시킨다. Grouping은 optional observed feature로 두며, partition mixture는 필요하지 않다.

비교군에는 같은 observed feature·supervision을 받는 3-class MLP를 둔다. Transition mask/temporal filtering을 제거해도 같은 성능이면 PGM structure의 이득으로 주장하지 않는다.

### 5.1 Confidence와 expected improvement는 다르다

현재 answer correctness belief b_t(R)가 높으면 confidence stop이 가능하다. b_t(R)가 낮더라도 future recovery probability가 낮으면 futility stop이 합리적일 수 있다. 이 둘을 별도로 보정·평가한다.

V2 continuation target은 t에서 min(t+4,cap)의 selector correctness 차이다. 이 target은 중간 시점에 종료할 수 있는 option value를 직접 모델링하지 않는다. 예를 들어 correctness가 0→0→1→1→0이면 4-step net gain은 0이지만 중간의 correct 시점에서 종료하는 정책에는 가치가 있다. 이는 설명용 예시이며 관측된 사례 수를 주장하지 않는다.

새 방향은 finite-horizon value recursion이다.

$$
V_t(D_t)=\max\left\{
p(\hat a_t=Y\mid D_t),
-\lambda\widehat c_{t+1}(D_t)
+\mathbb E[V_{t+1}(D_{t+1})\mid D_t]
\right\}.
$$

Cap에서 stop value만 남긴다. Conditioned expectation과 future observation model을 배워야 하며 b_t만으로 충분한 state라고 가정하지 않는다. Question-level cross-fitting으로 continuation target과 posterior의 in-sample optimism을 줄인다. Realized future correctness의 per-question maximum을 그대로 inference용 value인 것처럼 쓰면 clairvoyant oracle이 된다.

Bayesian optimal stopping 자체는 기존 아이디어다. 특히 최근 [Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](https://arxiv.org/abs/2602.05395)는 modal answer identification을 다룬다. 이 프로젝트는 ground-truth correctness와 recoverability를 supervision으로 삼는 차이를 명확히 해야 한다. 별도의 novelty 검토가 필요하다.

## 6. 후보 D: Partition 없는 pairwise factor graph

Hard clustering을 없애고 response pair의 relation을 직접 conditional factor에 넣을 수 있다.

$$
p(Y=y\mid D_t)\propto\exp\left[
b_t(y)+\sum_i u_i\mathbf1[a_i=y]
+\sum_{i<j}J_{ij}\mathbf1[a_i=y,a_j=y]
\right].
$$

High similarity의 redundant vote에는 negative J_ij를 줄 수 있다. Clustering의 transitivity나 threshold boundary를 강제하지 않는 장점이 있다. Candidate별 score를 계산해 normalize하므로 loop가 있는 correctness graph에 무조건 loopy BP를 돌릴 필요는 없다.

하지만 J가 constant여도 n_y(n_y−1)/2에 따른 generic count correction이 된다. 다음 control이 필수다.

- Constant J / count-only pair potential.
- Size/confidence-matched shuffled pair relations.
- Unary-only model과 같은 supervision·parameter budget.

V3/V4에서 relation의 추가 효과가 약했으므로 우선순위는 temporal state PGM보다 낮다. 이를 true error-dependence의 generative model로 해석하지 않는다.

## 7. 후보 E: Question-constraint factor graph — accuracy를 위한 정보 추가

Selection error 27개를 고치려면 answer frequency와 self-confidence 외의 정보가 필요할 수 있다. Candidate y가 question의 constraint를 만족하는지 계산한 factor를 추가한다.

$$
p(Y=y\mid D_t,Q)\propto
\phi_{\mathrm{SC}}(y;D_t)\prod_j\psi_j(y;Q).
$$

예를 들어 question에서 추출한 equation에 candidate를 대입해 residual을 계산한다. Unsupported constraint는 neutral factor로 두고 hard veto하지 않는다. Numeric agreement만 검사하는 것과 question premise를 검증하는 것은 다르다.

이 접근은 V3 question substitution coverage 0.4%를 실제로 확대할 parser/verifier가 있어야 의미가 있다. Coverage, parser precision, incorrect-candidate rejection, correct-candidate rejection을 모두 측정한다. Same verifier factor를 independent selector에도 제공해 verifier와 PGM의 효과를 분리한다. 아직 관측되지 않은 answer를 이 factor만으로 새로 만들어내지는 못한다.

## 8. 추천 순서와 연구 범위

| 우선순위 | 구성 | 기대하는 개선 | 주요 한계 |
|---|---|---|---|
| 1 | V2 positive support + single fixed rule | 단순화, mixture 필요성의 clean test | 이미 single control은 있음; accuracy 개선을 보장하지 않음 |
| 2 | Coverage–selection + three-state temporal PGM | Confidence/futility 구분, sampling allocation, intermediate stop | 같은 selector/cap의 timing oracle은 84.4% |
| 3 | Question-constraint factors | Observed correct candidate를 놓치는 selection error | 새로운 grounded information과 coverage가 필요 |
| 4 | Pairwise factor graph | Hard partition 없이 redundancy modeling | 현재 relation signal이 약해 generic scaling 반복 가능 |

61개 unseen-gold error를 크게 줄이려면 다른 reasoning strategy/prompt 또는 model로 candidate discovery를 바꾸는 action을 고려할 수 있다. 이는 현재 fixed-prompt bank만으로 평가할 수 없다. Action을 사전에 무작위 배정한 새 collection과 추가 generation/verifier 비용을 포함한 별도 실험이 필요하다.

가장 실용적인 다음 구성은 **single fixed partition selector + coverage/state PGM + honest continuation value**다. 다만 이를 바로 v2보다 낫다고 부르지 않고 fixed-budget selection, state prediction, adaptive token cost를 순서대로 분리해 검증한다. V2/V3/V4의 frozen artifact는 보존하며, 새 holdout은 설계를 고정한 뒤 사용한다.

---

## Historical source: `PGM_ASC_V3_DIAGNOSIS.md`

# V2 재분석과 v3 개선 가설

2026-11-01 KST. 아래 수치는 이미 평가한 v2 bank의 diagnosis이며, v3의 새 validation 결과가 아니다.

## 1. Stopping 개선과 PGM 기여를 분리해야 한다

V2 final PGM은 82.4% / 3,492.6 token, CGES는 82.6% / 3,032.4 token이었다.
Single partition도 82.4%였고, PGM−random calibrated NLL 차이는 +0.00113,
95% paired-bootstrap interval [-0.00310,+0.00596]이었다. Minimum/cap의 비용
절감이 equation/error-pattern grouping의 효과를 증명하지 않는다.

V3 primary endpoint를 **equation PGM vs matched random의 raw NLL 차이**로 둔다.
Single만 이기거나, calibrated confidence만 달라지는 결과도 구분해서 보고한다.

## 2. Candidate가 있어도 놓치는 selection gap이 남았다

| 고정 prefix | Gold가 관측된 비율 | PGM accuracy | Observed-candidate oracle gap |
|---:|---:|---:|---:|
| 8 | 89.6% | 82.6% | 7.0%p |
| 16 | 91.2% | 83.8% | 7.4%p |
| 32 | 92.6% | 83.4% | 9.2%p |

이는 달성 가능한 성능의 보장이 아니라, 관측한 answer 중 정답을 항상 고르는
oracle의 upper bound다. 더 생성하는 것만으로 이 gap이 닫히지 않았다.
Actual stopping의 wrong 88문제 중 27개는 이미 gold candidate가 있었다.
61개는 stop 시점에 gold가 없었으므로 selector 변경만으로 모두 해결할 수 없다.

V3에서는 response correctness BCE를 별도 supervision으로 추가한다. Input은
confidence와 observed math features이고, gold는 train target 생성에만 쓴다.
이 feature와 supervision을 independent·single·random control에도 똑같이 제공한다.
Outcome supervision 자체는 기존 verifier 연구의 아이디어이며, 새로운 PGM 기여로
주장하지 않는다. [Cobbe et al.](https://arxiv.org/abs/2110.14168)

## 3. Positive support만으로는 낮은 quality의 vote도 evidence가 된다

현재 positive score는 각 response를 softplus support로 더한다. V3는 learned
correctness probability q_i를 symmetric mismatch likelihood와 비교한다.

$$
r_i=\log\frac{q_i}{(1-q_i)/k}=\operatorname{logit}(q_i)+\log k.
$$

k는 현재 관측된 answer class 수다. 예를 들어 k=2, q=0.1이면 r<0이므로
그 response가 낸 answer를 약화할 수 있다. q가 정당하게 추정된다는 가정이
필요하며, BCE·calibration·independent control로 검증한다. Alpha=1, zero OTHER
bias, fixed q에서 CGES likelihood와 일치하는지 unit test로 확인했다.
[CGES](https://arxiv.org/abs/2511.02603)

동일 quality head를 positive score로 쓰는 control과 auxiliary BCE 없는 control을
두어, negative evidence와 supervision의 효과를 분리한다.

## 4. 새로운 grouping input이 필요하지만 semantic label을 꾸미면 안 된다

Hidden-state cosine의 높은 유사도만으로 shared wrong assumption을 알 수 없다.
V3는 literal equations와 number-masked structure의 TF-IDF/SVD representation을
추가하고, question의 공통 math 방향을 제거한다. Vocabulary·SVD·threshold는
train에서만 fit하고 response별로 transform하여 future leakage를 피한다.

이는 shared-error representation의 후보일 뿐이다. Same answer/both wrong을
common-error annotation으로 간주하지 않는다. Size-matched random control과
raw hidden-state control을 함께 두어 실제 추가 정보가 있는지 검사한다.

Arithmetic consistency와 제한된 question substitution은 **soft feature**다.
지원하지 않는 식은 missing이고, check 실패를 hard veto로 사용하지 않는다.
Coverage가 낮거나 systematic wrong assumption을 놓칠 가능성을 측정에 포함한다.

## 5. 이전보다 대표적인 development 데이터와 독립 검증이 필요하다

V2 calibration-A 150문제를 train에, calibration-B 300문제를 dev에 명시적으로
전환하여 train/dev를 950/500으로 확장한다. 모든 v3 arm에 같은 데이터를 제공한다.
V2 holdout 500문제는 학습에 넣지 않는다. 새 calibration 450 + holdout 500문제는
모든 이전 cohort와 normalized overlap 없이 고정했다. V3 holdout 생성·평가는
model/calibration lock 이후에만 시작한다.

따라서 frozen v2와의 차이에는 development 데이터 변경도 포함된다. Architecture
효과는 동일한 950/500 데이터로 학습한 v3 arm 사이에서 해석한다.

## 6. 실제 비용은 별도 endpoint로 남는다

V1 online timing은 PGM 평균 3,118.5초, CGES 897.4초였다. 이 값은 원래 정책의
100-question 공통 batch-8 serving 측정이며 v3 timing이 아니다. V3의 equation
feature도 추가 비용을 갖는다. Generated-token 절감만으로 latency 개선을 주장하지
않는다. 먼저 grouping/quality의 추가 효과를 확인하는 bounded replay experiment를
실행하고, 유용한 모델이 확인되면 serving 실험을 구성한다.

[V3 실행 계획](./PGM_ASC_V3_EXPERIMENT_PLAN.md) ·
[V2 원 지표](./artifacts/runs/gemma3_pgm_v2/evaluation.json) ·
[V1 online 측정](./artifacts/runs/gemma3_pgm_v1/online/report.md)

---

## Historical source: `PGM_ASC_V3_EXPERIMENT_PLAN.md`

# PGM-ASC v3: representation and response-quality experiments

Created 2026-11-01 KST before new v3 calibration or holdout response collection.

## Evidence and questions

V2 reduced generated-token cost mainly through minimum=2 and cap=8. Its final
PGM had accuracy 82.4% / 3,492.6 tokens, versus CGES 82.6% / 3,032.4 tokens on
the same 500 questions. Single and size-matched random controls remained close.
Of 88 PGM errors, 27 already had a correct candidate at stopping, and 61 did not.
Thus another stopping grid alone does not test the missing PGM-specific signal.

V3 separates three questions:

1. Does a train-only equation/step representation improve grouping compared to
   the original hidden-state representation, beyond size-matched random groups?
2. Does supervised response quality improve answer selection compared to
   confidence-only positive evidence, including without a partition mixture?
3. Does a quality-aware partition gate add value beyond geometry alone?

These are hypotheses, not established shared-error clusters. Answer equality and
both-wrong labels are not annotations of a common reasoning error. The new text
representation is a cheap observable proxy; no claim of semantic ground truth.

## Data and budget

Train: old v1 train 800 + v2 calibration-A 150 = 950 questions. Dev: old v1 dev
200 + v2 calibration-B 300 = 500. This explicitly transfers spent calibration
data into development to reduce the train/deployment difficulty mismatch. V2
holdout is diagnostic evidence only and is not added to train/dev. Every arm
uses exactly the same expanded development data. The frozen v2 model is a
historical reference, not a controlled architecture-only comparison.

Draw fresh v3 calibration-A 150, calibration-B 300 and holdout 500 from remaining
MATH test questions, excluding normalized overlap with every v1/v2 cohort.
Retain Asymptote questions as text. Seed 20261103; uniform without replacement.
LLM/model/data revisions and vLLM sampling remain pinned as in v2. N=32 means
30,400 new responses total. Keep gold separate from generation/extraction inputs.
Holdout generation/evaluation begins only after all model/policy locks are saved.

Use GPUs 4/5 for calibration banks in parallel, GPU 4 for small-head training,
and GPUs 4/5 for disjoint 250-question holdout shards. Existing other processes
are never stopped. Shared-device generation timing is not an isolated benchmark.

## Representation and observed features

Keep the raw hidden-state view. Add equation_residual: remove the last boxed
final-answer segment, extract math spans and equation lines, encode both literal
numeric expressions and a number-masked structure view using character 3–5 gram
TF-IDF (max 20,000 features), then train-only SVD to 64 dimensions. Subtract the
projection onto the question's math vector when available; normalize the
residual. Empty/zero vectors remain zero. Fit vocabulary, SVD, cosine thresholds
and standardizers on the 950 train questions only. Transform each response
individually; never center using future responses. Test representation changes
with the same confidence inputs and with size-matched online random controls.

Add small observed math features: math-span/step counts, simple numeric equality
checks, fraction/power/root counts, overlap of numbers with the question, numeric
answer availability, and conservative candidate substitution when the question
explicitly asks for one variable and supplies a supported equation. Unsupported
expressions are missing, not failed. The arithmetic evaluator is an AST
interpreter with bounded numeric operations, no eval, imports, or external tools.
These checks cover arithmetic and a restricted constraint class; they cannot
verify an entire derivation or its assumptions. Report coverage and failure rates.
All quality-aware controls receive identical features, including the non-PGM
and random controls. Do not give a verifier only to the proposed PGM.

## Conditional likelihood model

Learn a response correctness logit u_i from observed features, with
q_i = sigmoid(u_i). For k observed answer classes plus OTHER, a symmetric
mismatch likelihood has likelihood ratio

$$
r_i=\log\frac{q_i}{(1-q_i)/k}=u_i+\log k.
$$

For candidate partition m, use

$$
L_m(y)=\beta\sum_i n_{m,z_i}^{\alpha-1}r_i\mathbf1[a_i=y]
       +b(D_t)\mathbf1[y=\mathrm{OTHER}],
$$

and normalize over observed answers and OTHER. The common mismatch background
cancels in softmax. Invalid extracted answers contribute no vote. Positive
evidence controls replace r_i by softplus(u_i). beta is positive and alpha is
bounded to [0,1]. alpha=1 is grouping-independent and is an essential control.
This is a cluster-tempered conditional likelihood score, not proof of a correct
generative independence model. OTHER prior remains a learned prefix head.

Use the exact finite mixture sum_m pi_m softmax(L_m), with the same limited
online candidate partitions as v1. A quality gate adds eight observed summaries
of each partition's resulting confidence/evidence to its twelve geometry
features. Geometry-only and random counterparts remain separate controls.

Train question-balanced prefix NLL plus response correctness BCE with coefficient
1.0. Compute BCE once per complete response on train, rather than counting early
responses repeatedly in every prefix. Gold is used only for supervised targets.
The no-auxiliary arm has coefficient zero. This is final-answer outcome
supervision, not step correctness supervision or a full semantic verifier.

Outcome-supervised verification is established prior work, not v3 novelty:
[Cobbe et al.](https://arxiv.org/abs/2110.14168). The confidence likelihood control
is motivated by [CGES](https://arxiv.org/abs/2511.02603); learned quality and
partition discount must be evaluated separately from that existing idea.

## Fixed arms and training

1. positive/raw, base confidence features; positive/equation, identical features.
2. likelihood independent/base; likelihood independent/math features.
3. likelihood PGM/math/raw; likelihood PGM/math/equation.
4. likelihood size-matched random/math/equation.
5. likelihood single partition/math/equation, six prespecified candidates.
6. likelihood PGM/math/equation with quality gate and matched random quality gate.
7. likelihood PGM/math/equation without auxiliary BCE.
8. positive PGM/math/equation with the same auxiliary BCE.

There are 17 seed-42 training trials, including the six single candidates.
Architecture: two-layer 32-unit MLPs, AdamW lr=0.001, weight decay=0.001, at most
100 epochs, patience 15. Fixed initialization seed 42, question batches 32.
Prefix training gives equal mass to {2},{3–4},{5–8},{9–16},{17–32}; t=1 has no
prefix NLL weight because all deployed policies require two responses. Dev
selection uses equal mean NLL at t=2/4/8/16/32. No new learning-rate/width grid.

Select the primary equation PGM (geometry or quality gate) using dev NLL only.
Its primary random comparator uses the matched gate family. Select the single
candidate using dev only. Repeat the selected PGM, matched random and independent
math likelihood model for seeds 43/44 (six additional trials). Keep seed 42 as
the primary result; do not select the best seed or average independent questions
across seeds. Total: 23 fits, frozen LLM, no new LLM finetuning.

## Calibration, selection, evaluation

Cal-A fits a global temperature for each arm. Cal-B selects gamma from the v2
grid and cap from {8,16,32}, with min=2 and patience=1. No continuation head or
bucket-calibration grid in v3: the model mechanisms should remain interpretable.
Use the same policy search for CGES, ASC and ReASC; calibrate ReASC's original
confidence/gate on A. Report the fully frozen v2 model/policy as a separate
reference. Among candidates at least SC32-minus-1pp accurate on B, minimize
generated tokens; if none qualifies, mark infeasible and retain the most accurate.
Empirical feasibility does not establish a statistical guarantee.

Freeze all train/dev choices, calibration settings and artifact/code hashes
before evaluating the 500-question holdout. Primary mechanism endpoint: mean
uncalibrated NLL at t=2/4/8/16/32 for preselected equation PGM minus its matched
random comparator, paired question bootstrap (5,000 resamples). Also report
fixed-prefix accuracy/Brier, candidate-present selection gap, pairwise probability
TV/winner agreement, quality BCE/AUROC, arithmetic coverage, and seed variation.
Policy endpoints compare accuracy and generated tokens with CGES, independent,
single, random, SC8/16/32 and frozen v2. Report paired intervals, all arms,
Asymptote subgroup, failures/truncations and point-constraint failures.

Do not promote an arm based on holdout. Positive results against single without
a random-control improvement do not establish semantic grouping value. If none
improves, report the negative result and the useful non-PGM controls.

V3 is a mechanism/replay experiment. V2 and v3 end-to-end online latency remain
unverified; raw generation time and saved tokens are not substitutes. The older
v1 timing already shows a substantial feature-computation cost. Serving
optimization follows evidence of a useful model, rather than adding more graph
complexity before its value is established.

## Artifacts

- Config: `configs/pgm_v3.json`
- Modules: `pgm_pilot/pgm_v3_*.py`; driver `pgm_pilot/pgm_v3.py`
- Run: `artifacts/runs/gemma3_pgm_v3/`
- Live report: `progress.md`; final: `report.md`; locks never overwrite v1/v2.

---

## Historical source: `PGM_ASC_V4_EXPERIMENT_PLAN.md`

# PGM-ASC v4: grouping signal과 shared reliability 검증

작성: 2026-11-02 KST. 상태: **실험 계획이며 구현·학습·새 응답 수집은 아직 시작하지 않았다.**

## 1. 이번 실험의 결정

**기존 bank에서 grouping의 추가 가치를 먼저 검증하고, 통과한 경우에만 새로운 cap=8 adaptive 실험을 수행한다.**

V4의 primary question은 다음과 같다.

> 같은 response-quality probability와 sampling budget에서, reasoning-based grouping을 사용하는 shared-reliability PGM이 independent aggregation, generic evidence scaling, matched random grouping보다 좋은 answer distribution을 만드는가?

기존 v1/v2/v3의 model, code, cohort, lock은 수정하지 않는다. Operational reference는 frozen v2다. V4의 새 latent model은 검증할 가설이며 성능 개선을 전제하지 않는다.

| 순서 | 실험 | 새 LLM generation | 다음 단계로 넘어가는 조건 |
|---|---|---:|---|
| A | 기존 grouping의 evidence 경로와 control 재분석 | 0 | invariance·causality·matching audit 통과 |
| B | Frozen OOF quality + shared-reliability PGM 비교 | 0 | 아래 development gate 통과 |
| C | Fresh calibration과 1,000-question holdout | 최대 11,600 responses | model lock 후 calibration, policy lock 후 holdout |
| D | 실제 online timing | C가 실용적일 때만 | accuracy–token 결과와 실제 feature 비용을 함께 평가 |

이번 계획에서 continuation head, 새로운 LLM verifier, 새로운 math feature, learned partition gate, prompt 다양화는 추가하지 않는다. 각각 별도 component가 되어 grouping 실험의 해석을 어렵게 하기 때문이다.

## 2. V3에서 이어받는 근거

V3 primary PGM과 matched random의 raw mean NLL 차이는 +0.000720, 95% interval [-0.000688,+0.002165]였고, 최종 correctness vector도 500문제 모두 같았다. 같은 holdout에서 v3는 78.6% / 16,446.5 tokens, frozen v2는 77.6% / 3,365.5 tokens였다. V3의 +1.0%p 차이 interval은 [-0.8,+2.8]%p였다.

현재 model에서 partition은 주로 response contribution에 cluster-size discount를 곱한다. 따라서 다음을 분리해야 한다.

1. 어느 response를 함께 묶었기 때문에 좋아진 것인가?
2. 전체 evidence magnitude를 줄여 overconfidence를 완화했기 때문에 좋아진 것인가?
3. Quality head가 좋아진 것인가, aggregation이 좋아진 것인가?
4. Selector가 좋아진 것인가, threshold와 cap이 바뀐 것인가?

V3 numeric check coverage는 79.7%였지만 question substitution coverage는 0.4%였고 failure flag가 한 번도 켜지지 않았다. 이번에는 feature 수를 늘리는 대신 같은 quality input에서 aggregation mechanism을 검증한다. 새 representation의 필요성은 이 실험 결과로 판단한다.

근거: [v3 summary](./artifacts/runs/gemma3_pgm_v3/result_summary.md), [v2 summary](./artifacts/runs/gemma3_pgm_v2/result_summary.md).

## 3. 데이터 역할과 leakage 방지

### 3.1 기존 bank: development 전용

- Train 950: v1 train 800 + 이미 사용한 v2 calibration-A 150.
- Dev 500: v1 dev 200 + 이미 사용한 v2 calibration-B 300.
- 기존 32 responses/question을 재사용한다.
- Quality BCE는 train의 32 responses를 question-balanced하게 사용한다.
- Aggregation training은 t=2,...,8만 사용하며 {2}, {3–4}, {5–8}에 각각 같은 total weight를 준다.
- Model selection은 question별 t=2,4,8 NLL의 평균이다.
- 이미 확인한 v1/v2/v3 holdout은 diagnosis evidence일 뿐, 새로운 confirmatory test가 아니다. 이번 train/dev에 추가하지 않는다.
- 반복적으로 사용한 dev이므로 dev interval과 gate 통과를 독립적인 확증으로 해석하지 않는다.

### 3.2 Fresh cohort: B 통과 후에만 고정

| Cohort | Questions | Responses/question | New responses | 역할 |
|---|---:|---:|---:|---|
| Calibration-A | 150 | 8 | 1,200 | 각 method의 global temperature |
| Calibration-B | 300 | 8 | 2,400 | stopping threshold 선택 |
| Holdout | 1,000 | 8 | 8,000 | 한 번의 locked evaluation |
| 합계 | 1,450 | 8 | **11,600** | |

MATH test의 모든 기존 cohort와 NFKC + whitespace-normalized problem 중복을 제외하고 seed=20261104로 uniform sampling without replacement한다. V3 manifest상 3,550개 중 950개를 사용했으므로 약 2,600개가 남는 계산이며, 실행 시 전체 manifest와 실제 remaining pool을 다시 검증한다. 새로운 cohort의 ID를 지금 선택하거나 gold를 열지 않는다.

Asymptote 문제는 기존과 같은 text-only 조건으로 포함하고 subgroup을 보고한다. Gold는 generation/extraction input과 분리한다. Cohort manifest, preprocessing, checkpoint, policy, code hash를 각각 고정한다.

이번 fresh bank는 **8 responses까지만** 수집한다. 따라서 새 holdout에서 SC32 대비 accuracy 유지나 t=32 recovery를 주장하지 않는다. 이전 cohort의 SC32 수치를 새 cohort와 직접 비교하지 않는다.

## 4. Phase A: 제일 먼저 할 실험

**Fixed prefix t=2,4,8에서 기존 v3 evidence가 실제 grouping을 필요로 했는지 검사한다. 새 학습보다 이 분석을 먼저 수행한다.**

기존 seed-42 quality logits, alpha, OTHER head를 고정하고 실제 grouping, matched random, mass-only control을 비교한다. Gate 영향은 분리하여 (i) 기존 gate weight를 모든 counterfactual에 공통 적용, (ii) 공통 uniform weight를 적용한 결과를 각각 보고한다. 이는 기존 model에 대한 counterfactual diagnosis이며 v4의 정식 model 비교와 구분한다.

현재 partition m의 valid response weight를 다음과 같이 쓰자.

$$
d_{m,i}=n_{m,z_i}^{\alpha-1},\qquad
W_m=\sum_{i\in V_t}d_{m,i},\qquad
e_i=\operatorname{clip}(u_i,-8,8)+\log K_t.
$$

현재 answer score와 mass-only control은 각각 다음과 같다.

$$
S_m(y)=\sum_{i\in V_t}d_{m,i}e_i\mathbf1[a_i=y],
$$

$$
S_m^{\mathrm{mass}}(y)=\frac{W_m}{|V_t|}
\sum_{i\in V_t}e_i\mathbf1[a_i=y].
$$

Mass-only control은 같은 total discount weight를 사용하지만 어느 response가 discount를 받는지는 사용하지 않는다. W_m은 실제 grouping에서 얻으므로, 이를 feature 계산이 공짜인 독립 baseline으로 해석하지 않는다. V_t가 비어 있으면 observed-answer vote를 0으로 둔다.

보고할 항목:

- Raw / globally calibrated NLL, Brier, accuracy at 2/4/8.
- PGM–random 및 PGM–mass probability TV, winner disagreement.
- Gold-available rate와 그중 selector error rate.
- Cluster size 및 cluster × observed-answer count table.
- 실제 grouping이 winner를 바꾼 문제에서 corrected / regressed count.
- Same-question·same-answer 상태와 quality bin을 통제한 within/between-cluster correctness residual 차이. Question 단위로 집계한다. 이는 common-error의 semantic annotation이 아니다.

### 4.1 Random control 수정

V3는 real/random partition을 각자 deduplicate하여 prefix별 active candidate 수가 달라질 수 있었다. V4는 **고정된 6개 candidate slot을 양쪽 모두 유지하고 weight=1/6**으로 둔다. 중복 slot도 고정 prior의 일부로 유지한다. 이는 unique partition set에 대한 uniform distribution과 다르다.

- **R-size, primary:** 각 prefix에서 real assignment vector의 valid response index 사이를 permutation하여 cluster size와 valid-response count를 정확히 보존한다. Invalid response의 assignment는 그대로 둔다. 여섯 slot과 weight도 동일하게 유지한다.
- **R-answer, diagnostic:** 같은 observed answer를 낸 response 사이에서만 assignment를 permutation하여 cluster × answer count까지 보존한다.
- Randomness는 question ID, prefix, slot, training seed로 결정한다. Answer class는 normalized observed answer로 정의하며 extraction failure도 별도 관측 category로 처리한다.
- Prefix t의 random control은 현재 prefix 안에서 past assignment를 다시 섞을 수 있다. 미래 response는 사용하지 않는 causal prefix control이며, 과거 assignment를 영구 고정하는 original online clustering과는 다르다.
- R-answer는 answer-composition 경로를 통제한 mediation diagnostic이다. Grouping이 answer composition을 통해 효과를 낼 수 있으므로 R-answer를 이기지 못했다는 이유만으로 모든 grouping 효과를 부정하지 않는다.

## 5. Phase B: Quality와 dependency를 분리한 PGM

### 5.1 공통 response-quality head

모든 v4 learned arm에 같은 quality probability를 제공한다.

$$
q_i=p(c_i=1\mid x_i),\qquad c_i=\mathbf1[a_i=y^*].
$$

- Input: 기존 base confidence features 9개. V3 math features는 이번에 추가하지 않는다.
- Model: 9 → 32 GELU → 1, BCE only. Aggregation NLL과 joint training하지 않는다.
- Question-grouped 5-fold cross-fitting으로 train OOF q_i를 만든다. 같은 question의 response가 여러 fold에 걸리지 않는다.
- 각 fold의 scaler는 해당 fold의 training questions에서만 fit한다.
- Dev/fresh inference는 5개 fold model의 probability 평균을 사용한다. Train OOF에는 해당 question을 보지 않은 한 model을 사용하므로 ensemble variance 차이가 남는다. 이를 기록하며 모든 arm은 동일한 q_i를 받는다.
- q_i를 [1e-4, 1-1e-4]로 clip한다.
- AdamW lr=1e-3, weight decay=1e-3, batch=32 questions, 100 epochs를 고정한다. Fold-held-out label로 early stopping하거나 hyperparameter를 선택하지 않는다.
- Seeds 42/43/44 각각 cross-fit한다. Primary seed는 42로 고정하고 best seed를 선택하지 않는다.
- BCE, Brier, reliability diagram을 question 단위 uncertainty와 함께 보고한다. Response를 독립적인 statistical unit으로 세지 않는다.

Outcome-supervised response scoring 자체는 기존 아이디어다. 이번 contribution 후보는 같은 quality 정보를 받는 aggregation 사이의 차이다. [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

### 5.2 Shared-reliability latent variable

각 cluster g에 shared latent variable을 하나 둔다.

$$
U_g\sim\operatorname{Uniform}(0,1),\qquad
v_i(U_g)=(1-\rho)q_i+\rho\mathbf1[U_g\le q_i],
\quad i\in g.
$$

Conditional on U_g, 각 response의 correctness indicator를 Bernoulli(v_i)로 모델링한다. 중요한 성질은 다음과 같다.

$$
\mathbb E_{U_g}[v_i(U_g)]=q_i.
$$

즉, rho가 바뀌어도 **각 response의 marginal quality probability는 같다.** 같은 cluster의 두 correctness indicator에 대해 이 모델은 다음 covariance를 갖는다.

$$
\operatorname{Cov}(c_i,c_j\mid X,Z)
=\rho^2\left(\min(q_i,q_j)-q_iq_j\right),\quad i\ne j.
$$

- rho=0이면 cluster와 무관한 independent model이다.
- rho가 커지면 cluster 내부 correctness에 positive dependence를 부여한다.
- Singleton에서는 모든 rho에서 같은 one-response likelihood가 나온다.
- rho는 {0, 0.25, 0.5, 0.75, 0.95}에서 dev NLL로 선택한다. 그룹별 rho나 추가 gate는 학습하지 않는다.

이 구조는 개별 confidence를 바꾸지 않고 dependency만 바꾸기 위한 **제안된 conditional model**이다. 실제 LLM samples의 decoder-level dependency, true common-error mechanism, 또는 최적의 dependence family로 해석하지 않는다. Positive dependence만 표현하며, 잘못된 답의 종류별 shared confusion은 별도로 모델링하지 않는다.

### 5.3 Candidate likelihood와 exact finite integration

현재 observed answer 집합을 C_t, K_t=|C_t|로 두고 Y의 support는 C_t ∪ {OTHER}로 둔다. K_t≥1일 때:

$$
f_i(a_i\mid Y=y,U_g)=
\begin{cases}
v_i(U_g),&a_i=y,\\
\left(1-v_i(U_g)\right)/K_t,&a_i\ne y.
\end{cases}
$$

OTHER는 아직 관측하지 않은 correct answer를 묶은 상태이므로 모든 valid response가 mismatch다. Uniform mismatch는 CGES 계열과 연결되는 가정이며 실제 error distribution을 보장하지 않는다. [CGES](https://arxiv.org/abs/2511.02603)

Cluster factor는 shared variable을 적분하여 구한다.

$$
F_g(y)=\int_0^1\prod_{i\in g}f_i(a_i\mid y,u)\,du.
$$

이는 numerical quadrature 없이 계산할 수 있다. 해당 cluster의 q_i와 0,1을 정렬한 boundary를 b_0,...,b_L이라 하면 각 interval에서 모든 indicator가 일정하므로:

$$
F_g(y)=\sum_{j:b_{j+1}>b_j}(b_{j+1}-b_j)
\prod_{i\in g}f_i\left(a_i\mid y,\frac{b_j+b_{j+1}}2\right).
$$

Cluster당 최대 n_g+1개 interval이며 log-space logsumexp로 계산한다. 동률 boundary의 zero-width interval은 제외한다. Invalid extracted answer는 vote/likelihood에서 제외하되 failure flag는 prefix feature에 유지한다. Valid response가 없으면 answer return을 abstention/error로 기록하고 강제 종료는 cap에서만 한다.

공통 OTHER head b(w_t)를 포함한 partition별 distribution은:

$$
L_m(y)=b(w_t)\mathbf1[y=\mathrm{OTHER}]+\sum_{g\in z^{(m)}}\log F_g(y),
$$

$$
p(Y=y\mid D_t)=\frac16\sum_{m=1}^{6}\operatorname{softmax}(L_m)(y).
$$

OTHER head는 기존 w25 → 32 → 1 구조를 train OOF q_i의 independent likelihood에 대해 학습한 뒤, 해당 seed의 모든 arm에서 공통으로 freeze한다. Pure independence와 matched-random control을 포함해 같은 head를 사용한다. 이 고정 head가 다른 aggregation에 최적이지 않을 수 있으므로 gold가 관측된 prefix의 conditional NLL과 p(OTHER)를 함께 보고한다.

OTHER head와 I1의 optimization도 위와 같은 AdamW 설정, 100 epochs, 32-question batch를 사용한다. Dev를 보고 epoch 수를 늘리거나 추가 optimizer grid를 열지 않는다.

Candidate set과 feature는 response에서 구성되므로 이 식은 conditional scoring PGM이다. Free-form reasoning 전체의 joint generative likelihood나 candidate set이 바뀔 때의 exact Bayesian update를 주장하지 않는다. 제한된 6개 partition의 mixture일 뿐 전체 partition space를 적분하지 않는다.

### 5.4 고정 arm

| ID | Method | 검증 목적 |
|---|---|---|
| I0 | Independent likelihood, rho=0 | 정확히 nested된 control |
| I1 | Independent likelihood + generic evidence scale | grouping 없이 confidence magnitude만 조정 |
| S0 | 기존 cluster-size discount, frozen q_i | 새 latent structure와 기존 discount 비교 |
| H0 | Shared reliability + real equation partition mixture | **Primary v4 PGM** |
| H1 | Shared reliability + R-size mixture | **Primary matched random** |
| H2 | Shared reliability + all-one partition | question-wide shared difficulty control |
| H3 | Shared reliability + observed-answer partition | answer agreement만으로 가능한 aggregation |
| H4 | Shared reliability + dev-selected single real partition | partition mixture의 추가 가치 |
| H5 | Shared reliability + R-answer mixture | answer composition을 통제한 diagnostic |

Equation encoder와 thresholds는 기존 v3 train-only artifact를 고정 재사용한다. Encoder를 새 dev/fresh data에 다시 fit하지 않는다. H4는 기존 여섯 candidate slot과 rho grid에서 dev로 한 조합만 선택한다. S0의 alpha는 {0,0.25,0.5,0.75,1}에서 선택하며 positive/negative score 선택을 추가하지 않고 기존 likelihood evidence를 사용한다.

I1의 score는 I0의 observed-answer log-likelihood ratio에 positive scale beta_t를 곱한다. beta_t=exp(clip(a^T h_t,-3,3)), h_t=[1,log(t),normalized vote entropy,mean(q_i)]이며 coefficient 네 개만 train prefix NLL로 학습한다. 공통 OTHER bias는 유지한다. 이 control은 partition assignment를 사용하지 않는다.

각 arm은 같은 q_i, OTHER head, prefix weight, train/dev questions를 사용한다. R-size와 R-answer는 같은 rho grid를 갖는다. Fresh data에서는 이미 선택한 rho/alpha/single slot을 바꾸지 않는다.

### 5.5 Development gate

새 generation에 들어가는 조건은 다음과 같다. 이는 dev 기반 자원 배분 규칙이며 statistical confirmation이 아니다.

1. Seed 42 H0의 mean NLL(2/4/8)이 H1보다 최소 **0.005 nats** 낮다.
2. Seed 42 H0가 dev에서 고른 strongest non-reasoning control C*보다 최소 **0.005 nats** 낮다. C*는 I0/I1/H2/H3 중 dev NLL 최소인 하나로 고정한다.
3. H0의 accuracy@8이 H1과 C* 각각보다 1%p 넘게 낮지 않다.
4. Seeds 42/43/44 중 적어도 두 seed에서 H0−H1 및 H0−C* NLL 차이의 방향이 모두 음수다.
5. Causality, matched-size/slot, marginal-quality invariance와 numerical audit가 통과한다.

C*의 method ID는 seed 42 dev에서 한 번 선택하여 extra seed에도 유지한다. 각 seed의 rho/alpha는 같은 사전 grid로 선택하며, test를 보고 seed나 comparator를 교체하지 않는다.

0.005 nats는 사전에 정하는 practical screening threshold다. Gate가 실패하면 fresh cohort를 사용하지 않고 결과를 남긴다. Budget 내에서 새로운 architecture/HPO를 연쇄 추가하지 않는다. H4를 이기는지는 별도 hypothesis이며, H0가 H4를 못 이기면 partition-mixture의 필요성은 주장하지 않는다.

## 6. Phase C: Budget를 고정한 adaptive 비교

### 6.1 공통 policy

- Minimum responses=2, maximum responses=8, patience=1.
- Calibration-A: method별 global temperature 하나만 fit한다. Temperature 뒤에 rho, alpha, q_i, partition을 다시 fit하지 않는다.
- Calibration-B: gamma ∈ {0.5,0.6,0.7,0.8,0.85,0.9,0.925,0.95,0.975,0.99,0.995,never-stop}에서 선택한다.
- Stop은 observed answer의 calibrated probability가 gamma 이상인 경우에만 허용한다. OTHER가 가장 크면 계속 생성하고 cap에서는 최고 observed answer를 반환한다.
- Cal-B에서 accuracy ≥ max(accuracy_SC8,accuracy_frozen_v2)−1%p인 정책 중 평균 generated token이 가장 작은 것을 선택한다. Tie는 높은 accuracy, 적은 responses, 높은 gamma 순서로 해소한다.
- Feasible policy가 없으면 infeasible로 표시하고 accuracy 최대, token 최소 순으로 diagnostic policy를 남긴다. Accuracy constraint를 사후 완화하지 않는다.
- 각 control도 같은 policy search를 받는다. Original method 고유 gate가 필요한 ReASC는 A에서 먼저 해당 gate/normalization을 고정한다.

Frozen v2는 checkpoint, scaler, temperature, min=2/cap=8, gamma와 continuation head를 그대로 적용하는 historical reference다. Frozen v2는 v4와 training data/objective가 다르므로 architecture-only control이 아니다. 기존 feature extraction을 같은 새 bank에서 수행하되 모든 TF overhead를 기록한다.

### 6.2 보고할 method

- Phase B의 9개 arm 전체. Primary model은 H0 seed 42이며 test를 보고 바꾸지 않는다.
- Frozen v2, 동일한 cap/min 조건으로 보정한 CGES, ASC, ReASC.
- SC2/SC4/SC8. Reused old cohort의 SC32와 섞지 않는다.
- Seeds 43/44는 robustness로 별도 보고한다. 같은 1,000 questions를 3,000 independent observations로 세지 않는다.

### 6.3 Endpoint와 성공 판정

**Mechanism endpoint:** question별 mean raw NLL at t=2,4,8.

Fresh test에서 H0−H1, H0−C*, H0−H4의 세 contrast를 고정한다. Question-paired bootstrap 10,000회와 각 contrast의 **98.33% interval**을 보고하여 세 비교의 Bonferroni familywise level을 약 5%로 맞춘다. Descriptive 95% interval도 별도로 제공한다.

- Reasoning grouping claim: H0가 H1과 C* 각각에 대해 point NLL ≥0.005 nats 개선이고 corrected interval upper bound <0이어야 한다.
- Partition mixture claim: 추가로 H0−H4 corrected interval upper bound <0이어야 한다.
- Calibrated NLL/Brier, fixed-prefix accuracy, winner disagreement, TV, p(OTHER), gold-available selection gap은 secondary endpoint다.
- Lower NLL만 좋아지고 fixed-prefix answer accuracy가 같으면 confidence 개선으로 보고하고 answer-selection 개선과 구분한다.

**Deployment endpoint:** locked stopping policy의 accuracy와 generated token, frozen v2 대비 question-paired 차이.

- Practical point target: accuracy ≥ frozen v2−1%p, token ≤ frozen v2의 90%.
- Statistical accuracy 유지 claim은 accuracy difference의 95% lower bound >−1%p인 경우에만 한다.
- Token 개선 claim은 paired token difference의 95% upper bound <0인 경우에만 한다.
- 한 endpoint만 통과하면 두 조건의 동시 개선을 주장하지 않는다.
- 1,000 questions가 1%p non-inferiority를 보장하지 않는다. 예를 들어 paired discordance가 6%라면 normal approximation의 95% half-width는 약 1.52%p다. 표본 수는 고정하고 결과가 애매하다고 holdout을 추가로 모으지 않는다.

전체 error는 (i) stop 시 gold available, (ii) stop 시 gold absent, (iii) cap exhausted로 나눠 보고한다. Recovery는 이번 bank에서 관측 가능한 t≤8까지만 계산한다. Extraction failure/truncation은 삭제하지 않고 method별 failure behavior와 subgroup을 기록한다.

## 7. Phase D: 실제 비용 확인

C에서 practical target을 통과한 경우, 미리 정한 fresh holdout의 100-question subset에서 frozen v2, H0, C*의 online serving을 측정한다. Subset은 model 성능을 보지 않고 question hash 순으로 선택한다.

- 공통 batch=8, active question당 한 response씩 생성하는 round-based scheduler.
- 같은 question의 3회 repeat, rotating method order, matched decoding seed schedule.
- Generation, feature TF, quality/partition/factor inference, end-to-end time을 분리한다.
- 같은 scheduler·hardware condition에서 비교하고 GPU utilization 및 다른 process 존재를 기록한다.
- TF를 필요한 method에만 실행하고, 다른 method에도 사용했다면 그 비용을 명시한다.
- GPU 4/5 사용은 기존 authorization 범위이나 다른 process를 중단하지 않는다. 여유가 없으면 이 측정만 pending으로 둔다.
- Shared-device timing과 100-question 반복을 독립 300문제의 accuracy evidence로 해석하지 않는다.

## 8. 고정 generation 설정과 예산

- LLM: google/gemma-3-4b-it, revision 093f9f388b31de276ce2de164bdc2081324b9767.
- Dataset: EleutherAI/hendrycks_math, revision 21a5633873b6a120296cce3e2df9d5550074f4a3.
- vLLM 기존 environment, BF16, temperature=0.7, top_p=0.95, top_k=0.
- max_new_tokens=4096, max_model_len=8192, max_num_seqs=128.
- Prompt: 기존 solve-step-by-step + boxed final answer, 응답 간 prompt history 공유 없음.
- Fresh samples/question=8. Train/development의 기존 32-response bank와 구분한다.
- Phase A/B 새 generation=0; Phase C 최대 11,600 responses. Online repeat generation은 별도 비용이다.
- Quality training은 5 folds × 3 seeds=15 fits. OTHER head 3 fits, I1 scale 3 fits로 총 21개 작은 optimization run이다. Epoch/model grid는 추가하지 않는다.
- rho/alpha/single-slot 비교는 frozen feature에서의 scoring grid이며 별도 LLM 또는 neural head fit이 아니다.

## 9. 구현 순서, 검증과 산출물

다음 경로는 **구현 예정**이며 현재 존재하는 module로 간주하지 않는다.

1. `pgm_pilot/pgm_v4_controls.py`: mass-only, fixed-slot R-size/R-answer, causal audit.
2. `pgm_pilot/pgm_v4_quality.py`: question-grouped OOF quality와 공통 OTHER head.
3. `pgm_pilot/pgm_v4_model.py`: shared-reliability factor의 exact finite sum, rho=0 control.
4. `pgm_pilot/pgm_v4_develop.py`: 고정 9-arm 비교와 development gate 기록.
5. `pgm_pilot/pgm_v4.py`, `pgm_v4_evaluate.py`: gate 통과 시 cohort, calibration, lock, single holdout evaluation.

필수 검증:

- 각 response의 integrated quality가 q_i와 수치적으로 일치한다.
- rho=0에서 모든 partition/control의 probability가 I0와 일치한다.
- Singleton factor는 모든 rho에서 independent response likelihood와 일치한다.
- n≤3 toy case에서 exact interval sum과 고해상도 numerical integration이 일치한다.
- Future response 변경이 과거 prefix feature/partition/prediction을 바꾸지 않는다.
- R-size는 prefix·slot별 size multiset과 valid-response count, R-answer는 추가로 cluster × answer count를 보존한다.
- 모든 mixture의 slot 수와 weight가 같고 permutation에서 대응하는 ID 재명명이 결과를 바꾸지 않는다.
- Quality-fold question overlap=0, inference에 gold가 들어가지 않는다.
- Invalid answers, empty valid set, duplicated q_i, tiny interval에서 finite output과 명시적 fallback을 확인한다.

산출물은 `artifacts/runs/gemma3_pgm_v4/` 아래 `control_audit.md`, `quality_oof_report.md`, `development_report.md`, `development_gate.json`으로 시작한다. Gate 통과 시에만 `cohort_lock.json`, `model_lock.json`, `selection_lock.json`, `evaluation.json`, `result_summary.md`, accuracy–cost figure를 추가한다.

## 10. 결과별 다음 결정

| 결과 | 해석과 다음 결정 |
|---|---|
| Mass-only / I1과 같음 | 현재 개선은 generic evidence scaling으로 설명 가능. 새 grouping 복잡도를 추가하지 않는다. |
| Shared reliability가 random에서도 같음 | Aggregation의 generic dependence 효과. Reasoning grouping contribution은 없다. |
| Real grouping이 random/C*를 이기고 single과 같음 | Grouping signal은 있으나 partition mixture의 필요성은 미확보. |
| NLL만 개선, token/accuracy target 실패 | Confidence modeling 결과로 보고하고 v2를 유지한다. |
| Mechanism과 practical target 모두 통과 | Online timing으로 total cost를 확인한다. |
| Phase B gate 실패 | V4를 negative result로 마친다. 별도 후속 계획에서 question-grounded representation 또는 실제 error annotation의 필요성을 다룬다. |

Aggregation에서 confidence와 shared reliability를 다루는 발상 자체는 novelty 주장이 아니다. 기존 대규모 answer-space aggregation 연구와의 관계도 검토해야 한다. [Titov et al., AISTATS 2010](https://proceedings.mlr.press/v9/titov10a.html). 위 latent construction은 이번 실험을 위한 제안이며 우수성이나 문헌상 최초성을 확인한 상태가 아니다.

[기계 판독용 계획](./configs/pgm_v4_plan.json)

---

## Historical source: `PGM_ASC_V5_EXPERIMENT_PLAN.md`

# V5: frozen-v2 single partition와 temporal decision PGM

2026-11-02 KST. 사용자가 승인한 single fixed partition + three-state PGM + honest continuation 실험의 실행 계획이다. 기존 frozen artifact는 수정하지 않는다.

## 질문과 범위

1. 같은 v2 positive-support/OTHER head에서 partition mixture를 고정 single rule로 바꿔도 selection과 calibrated stopping이 유지되는가?
2. 같은 observed feature와 supervision에서 U/W/R temporal PGM이 feedforward state classifier와 unconstrained chain보다 좋은 state prediction을 만드는가?
3. Fixed-4-step continuation 대신 intermediate stopping을 고려한 finite-horizon policy가 accuracy–token tradeoff를 개선하는가?

Question-constraint verifier, pairwise relation graph, prompt 변경은 이번 범위에 추가하지 않는다. 이 실험은 위 세 질문을 분리하며 reasoning-grouping의 semantic superiority나 Bayesian optimality를 전제하지 않는다.

## 데이터와 고정 selector

Frozen selector checkpoint는 v2 `refits/mixture/checkpoint.pt`, 기존 v1 final_full representation/scaler를 사용한다. Support/OTHER/alpha를 다시 학습하지 않는다.

- **M:** 원래 partition mixture.
- **S:** 같은 checkpoint, global fixed slot=1, eta≈0.96559975. Primary selector다.
- **A/I:** 같은 checkpoint의 all-one/singleton, fixed-budget 및 threshold diagnostic.
- **N:** 기존 native v2 single checkpoint. 별도로 학습된 historical single control이며 clean same-head ablation과 구분한다.

State/policy train **900 questions** = v2 calibration-A/B 450 + v3 calibration-A/B 450. 이 900문제는 frozen v2 support head의 800-question 학습과 normalized overlap이 없어야 한다. Frozen v2의 기존 calibrator/continuation에는 일부 사용됐지만 새 state/policy는 raw selector distribution을 input으로 사용한다.

Dev **700 questions** = v1 dev 200 + **spent v2 holdout 500을 development로 명시 전환**. 이 500문제의 종전 결과는 v5의 독립 검증이 아니며 설계·선택에 사용된다. V3 holdout은 새 training/dev에 추가하지 않는다. 모든 모델에 같은 train/dev를 제공한다. 기존 bank의 첫 8 responses만 사용한다.

Fresh calibration-A 150, B 300, holdout 1,000을 모든 기존 cohort와 normalized overlap 없이 seed=20261105로 선택한다. 문제당 8 responses, 총 **11,600**. Gold는 입력과 분리한다. Model lock 전 fresh calibration을 열지 않고, policy lock 전 holdout을 생성·평가하지 않는다. Cohort manifest를 고정한 뒤 holdout을 늘리지 않는다.

V4의 grouping-specific NLL gate는 다른 실험의 기준이었다. V5는 negative result도 포함한 clean single/temporal 비교 자체가 목적이므로, 개발 결과가 미미하더라도 correctness·causality·OOF audit를 통과하면 사전 지정한 fresh 평가를 수행한다. Dev로 primary model을 바꾸지 않는다.

## State prediction

U=gold candidate unseen, W=gold seen/current selected candidate wrong, R=current selected candidate correct. State label은 training/evaluation target에만 사용한다. Inference 함수는 gold를 받지 않는다.

입력은 기존 observed w25, selector의 top/runner/OTHER/margin/conditional entropy, new-answer flag, selected-answer change, stable-run fraction, last-response token length 및 base9 confidence다. 모든 variant가 같은 입력을 받는다. Scaler는 각 model의 train questions에서만 fit한다.

- `mlp`: 43→32 GELU→3.
- `unmasked`: 같은 32-unit trunk에서 emission 3 + transition 9. Conditional chain, transition restriction 없음.
- `masked`: 같은 chain에 다음 논리적 mask를 적용. **Primary graph**.

W/R→U 금지. New answer가 없으면 U→W/R 금지. Selected answer identity가 같으면 W↔R 금지. 모든 모델에 공통으로 K=0이면 U만, K=1이면 U/R만 허용하는 point-feasibility mask를 적용한다. 이는 trivial feasibility를 temporal model만의 이득으로 만들지 않기 위함이다.

$$
b_t(s)\propto\exp e_\theta(s,o_t)
\sum_{s'}T_\theta(s',s\mid o_t)b_{t-1}(s').
$$

Prefix summary를 쓰는 conditional filtering model이며 iid emission의 generative likelihood를 주장하지 않는다. Hidden coverage state의 monotonicity를 posterior U probability의 monotonicity와 혼동하지 않는다.

Training: AdamW lr=0.003, weight_decay=0.001, 100 fixed epochs, question batch 128, gradient clip=1. t=1의 NLL weight는 0, {2},{3–4},{5–8}에 같은 total weight. Fold-held-out label로 early stopping하지 않는다. MLP는 chain보다 transition-output parameter가 적으므로 parameter count를 함께 보고하며, same-parameter unmasked chain이 구조 효과의 핵심 control이다.

Seed 42: M/S × 3 graph variants=6 arms. Seeds 43/44: S × 3 variants=6 extra arms. 각 arm은 5 question folds + train900 full fit을 실행한다. 총 **72 neural fits**. Full model은 dev/fresh inference에 사용한다. OOF state prediction과 전체 OOF policy evaluation을 별도 저장하며 같은 question을 independent prefix들로 split하지 않는다.

## Continuation과 policy

Min=2, cap=8. 다음 세 family를 모든 state arm에 제공한다.

1. **Threshold:** state P(R)의 calibrated probability에 gamma 적용.
2. **Fixed4:** 동일 observed/state feature의 Ridge(alpha=10)가 min(t+4,8)의 selected-answer correctness minus current correctness와 추가 token을 예측. 예측 gain≤lambda×예측 token이면 futility stop. Confidence stop도 같은 gamma grid로 제공한다.
3. **Bellman/LSMC:** cap에서 시작하여 backward로 stage-specific Ridge(alpha=10)를 학습한다. Target은 다음 단계 이후 학습한 policy가 선택한 stop의 realized correctness와 future token cost다. Stop/continue는 predicted expectation과 model-estimated current P(R)로 결정한다. Per-question realized future correctness maximum을 decision input 또는 value target으로 사용하지 않는다.

Bellman의 backward training에는 train trajectories와 그 안에서 모델이 선택한 payoff가 쓰인다. Inner-training fitted decision의 optimism이 남을 수 있으므로 **전체 pipeline을 outer question fold 밖에서 평가**한다. Outer held question의 label은 state model, scaler, value regression, policy fitting에 전혀 사용하지 않는다. In-sample state probability를 value fitting에 쓰는 각 outer/full pipeline의 제한을 명시한다. 전체 OOF policy 성능을 final fresh 성능으로 취급하지 않는다.

여기서 OOF isolation은 새로 학습하는 state/value pipeline에 적용된다. Frozen v2 support weight 학습과 train900은 분리되지만, v2의 과거 stage/calibration 선택에는 train900 중 v2 calibration 450이 사용됐다. 역사적 selector 선택까지 완전히 OOF였다고 주장하지 않는다. 새로운 최종 holdout은 이 모든 데이터와 분리한다.

Value input은 동일 observed features와 current state probability 3개. Confidence calibration은 confidence stop에만 적용하며 value family의 raw reward representation을 뒤늦게 바꾸지 않는다.

Lambda grid={0,1e-6,3e-6,1e-5,3e-5}; gamma grid={0.5,0.6,0.7,0.8,0.85,0.9,0.925,0.95,0.975,0.99,0.995,never}. Calibration-A에서 state distribution의 global temperature와 selector answer distribution의 global temperature를 각각 fit한다. B에서 family/gamma/lambda를 선택한다.

B feasibility는 accuracy≥max(SC8,frozen-v2)−1%p, feasible 중 mean generated token 최소. 실패하면 infeasible 표시 후 accuracy 최대/cost 최소의 diagnostic policy를 고정한다. Holdout을 보고 family·threshold·lambda·selector를 바꾸지 않는다. Frozen v2 원래 checkpoint/calibrator/continuation/policy는 별도 untouched reference다.

## Endpoints

Primary는 S-masked seed42의 locked adaptive policy다. 같은 S의 mlp/unmasked는 matched controls, M-masked는 mixture comparison이다. 이 이름들은 dev/B의 winner로 교체하지 않는다.

- Clean selector endpoint: fixed-budget t=2/4/8 answer NLL, accuracy, probability TV/winner agreement, M−S paired contrast.
- State mechanism: mean NLL at t=2/4/8 및 U/W/R Brier, coverage/correctness calibration. S-masked−S-mlp 및 S-masked−S-unmasked의 97.5% question-bootstrap interval을 제시한다(2 contrasts Bonferroni). Seed 43/44는 같은 questions의 robustness이며 독립 표본으로 합치지 않는다.
- Policy: accuracy, responses, generated tokens, confidence/futility/cap error decomposition, observed/absent-gold error. Threshold/fixed4/Bellman의 선택과 전체 grid를 공개한다.
- Frozen v2 대비 practical target은 accuracy 하락≤1%p, token≥10% 절감. Statistical non-inferiority는 paired accuracy interval lower bound>−1%p일 때만 주장한다. 1,000문제가 이를 보장하지 않는다.
- No new SC32 comparison. V5 cap=8에서의 결과를 이전 cohort의 SC32 수치와 섞지 않는다.

CGES/ASC/ReASC는 동일 fresh bank와 min/cap 조건에서 B로 정책을 선택한다. CGES의 pinned confidence definition, ReASC의 A gate/normalization을 보존한다. Generated-token replay에는 TF/feature/head 비용이 포함되지 않으며 latency claim을 하지 않는다. 유용한 결과가 있으면 별도 online timing을 구성한다.

## 수집과 산출물

Gemma-3-4b-it 및 MATH revision, prompt, decoding, BF16, max_new_tokens=4096, max_model_len=8192는 기존과 같다. vLLM은 GPU 4/5를 사용하되 다른 process를 중단하지 않는다. GPU memory utilization=0.45로 두어 shared-device 여유를 남긴다. 8 responses/question, sampling seed42. TF와 supplemental baseline confidence를 같은 token IDs에서 추출한다.

새 파일은 `pgm_v5_*.py`, `configs/pgm_v5.json`, `artifacts/runs/gemma3_pgm_v5/`, `artifacts/data/math_pgm_v5/`에 둔다. Preflight tests, whole-pipeline OOF audit, old/new lock hash를 기록한다. State model/code/preprocessing lock→calibration bank→policy lock→holdout bank→single evaluation 순서를 지킨다. Restart는 동일 artifact/config에서만 허용한다.

---

## Historical source: `PGM_ASC_improvement_analysis.md`

# PGM-ASC: 개선 우선순위와 근거

작성일: 2026-11-01 KST. 범위: 현재 구현·고정된 MATH-500 결과·기존 bank의 post-hoc diagnostics. 아래 신규 policy 수치는 독립적인 새 validation 결과가 아니다. 원래 primary 결과와 실행 중인 online control policy는 별도로 유지한다.

- [Primary result summary](./artifacts/runs/gemma3_pgm_v1/test/analysis/result_summary.md)
- [Mechanism diagnostics](./artifacts/runs/gemma3_pgm_v1/test/analysis/improvement_diagnostics/report.md)
- [Machine-readable diagnostics](./artifacts/runs/gemma3_pgm_v1/test/analysis/improvement_diagnostics/evaluation.json)
- [Minimum-sample ablation](./artifacts/runs/gemma3_pgm_v1/test/analysis/min_samples_ablation/report.md)

![Post-hoc diagnostics](./artifacts/runs/gemma3_pgm_v1/test/analysis/improvement_diagnostics/overview.png)

위 accuracy-cost 점들은 point estimate다. 독립적인 새 validation이나 비열등성 proof를 나타내지 않는다.

## 1. 관측 결과에서 출발하기

| 관측 | 현재 값 | 해석 범위 |
|---|---:|---|
| PGM actual stopping accuracy | 79.4% | Original min=1, gamma=0.85 |
| PGM fixed-budget accuracy@32 | 83.0% | 같은 selector에 32개를 모두 제공 |
| SC32 accuracy | 82.4% | PGM selector의 차이는 3문제 |
| t=1 stopping | 318/500 = 63.6% | 해당 prefix에는 partition 구분이 없음 |
| t=1 incorrect stopping | 22문제 | Warm-up 효과가 발생하는 주요 원인 중 하나 |
| Recoverable premature error | 19문제 | 같은 selector를 N=32까지 실행하면 correct가 되는 early error |
| Budget exhaustion | 98/500문제 | 이 중 60문제가 incorrect |
| Exhausted questions의 token 비중 | 87.0% | Cost가 긴 tail에 집중 |
| PGM vs size-matched random grouping answer agreement@32 | 99.4% | 현재 reasoning grouping의 추가 효과가 약함 |

PGM fixed-budget 83.0%와 early-stop 79.4%의 차이에는 19개의 recoverable error와, early correct이지만 final incorrect인 반대 방향 1개가 포함된다. 따라서 accuracy 저하를 전부 answer selection 문제로 볼 수 없다.

## 2. 우선순위 1: continuation budget을 별도 decision으로 학습

현재 policy는 높은 confidence에서 early stop하고, 그렇지 않으면 최대 32개까지 계속한다. 낮은 confidence만으로 추가 sampling의 가치를 알 수는 없다.

98개 exhausted question은 전체 생성 token의 87.0%를 사용했다. 이 집단에서 t=16에는 39개가 correct였고, t=32에는 38개가 correct였다. 16→32 사이 3개가 correct로 바뀌고 4개가 incorrect로 바뀌었다. 이것이 모든 late sampling이 불필요하다는 뜻은 아니지만, 전체적으로 매우 비싼 continuation이었다.

기존 모델·T·gamma를 고정한 post-hoc replay는 다음과 같다.

| min_samples | patience | max_samples | Accuracy | Mean responses | Mean tokens |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 32 | 81.6% | 9.068 | 13,198.8 |
| 2 | 1 | 16 | 81.8% | 5.700 | 7,629.1 |
| 4 | 1 | 32 | 82.2% | 10.796 | 14,635.7 |
| 4 | 1 | 16 | 82.4% | 7.264 | 8,765.4 |
| 4 | 1 | 8 | 81.6% | 5.246 | 5,508.3 |

`min=4, cap=16`은 이 bank에서 SC32와 같은 accuracy 점 추정으로 약 69.4% 적은 생성 token을 사용했다. 이는 test를 본 뒤 얻은 후보이며 일반화·통계적 비열등성의 증명이 아니다.

가장 작은 후속 변경은 cap=16과 cap=32를 calibration에서 비교하는 것이다. 더 연구적인 확장은 continuation-value head다.

$$
\Delta_t=\mathbb E\left[\mathbf1\{A(D_{t+\delta})=Y\}-\mathbf1\{A(D_t)=Y\}\mid D_t\right],
$$

$$
\text{continue if }\widehat\Delta_t>\lambda\,\widehat{\Delta\mathrm{tokens}}_t.
$$

Train bank의 future prefix를 supervised target 생성에만 사용하고, inference input에는 D_t만 넣는다. 추가 sampling으로 correct이 되는 경우와 incorrect로 바뀌는 경우를 모두 target에 반영한다. 고정 cap, same-confidence policy, learned non-PGM continuation head와 비교해 PGM 특유의 추가 효과를 구분한다. Low-value stop도 동일하게 accuracy에 포함하며 실패한 질문을 제외하지 않는다.

## 3. 우선순위 2: prefix-aware calibration과 stop-aware objective

현재 train loss는 1..32 prefix를 균등하게 평균하고, dev hyperparameter 선택은 t=8/16/32 mean NLL을 사용한다. 반면 original policy는 63.6%의 질문을 t=1에서 끝냈다. 실제 decision이 집중되는 구간과 model selection 기준 사이에 차이가 있다.

후속 실험에서는 minimum 이후의 early prefixes를 포함한 prefix-bucket loss와 dev score를 사전에 정의한다. 예를 들어 min=2라면 2,3–4,5–8,9–16,17–32 bucket에 같은 총 가중치를 줄 수 있다. 더 복잡한 stop-time weighting은 train의 question-level OOF policy로 추정하며 test stopping frequency로 학습하지 않는다.

현재 OTHER calibration도 prefix와 domain에 따라 다르다.

| Prefix | Cal-B actual unseen | Cal-B mean predicted OTHER | Test actual unseen | Test mean predicted OTHER |
|---:|---:|---:|---:|---:|
| 1 | 16.7% | 16.7% | 26.8% | 21.0% |
| 4 | 9.3% | 10.2% | 15.0% | 13.1% |
| 16 | 7.3% | 6.6% | 9.4% | 8.5% |
| 32 | 6.0% | 6.8% | 7.2% | 9.3% |

초기 prefix에서는 test의 unseen answer 위험을 낮게 예측하고, 마지막 prefix에서는 평균 OTHER가 실제보다 높다. 모든 t에 같은 T를 적용하는 현재 방식에서 남아 있는 차이다. 이 평균 차이만으로 개별 질문의 calibration을 판정하지 않는다.

후속 후보는 적은 수의 prefix bucket별 T와 OTHER offset이다.

$$
\widetilde p_t(y)=\operatorname{softmax}_y\left(\frac{\log p_t(y)+b_{b(t)}\mathbf1[y=\mathrm{OTHER}]}{T_{b(t)}}\right).
$$

파라미터 수를 작게 두고 global T로 shrink하는 regularization을 사용한다. Calibration-A에서 fit하고 calibration-B에서 전체 policy를 검증한다. Cal-B의 single-response accuracy는 83.3%, test는 73.2%이므로 prefix calibration만으로 domain shift가 해결된다고 가정하지 않는다. 대표성 있는 calibration cohort도 필요하다.

PGM single-response ranking AUROC는 test에서 0.901, bottom-10% self-certainty는 0.875였다. 따라서 현재 confidence ranking이 전반적으로 약하다고 단정할 근거는 없다. 높은 coverage에서의 threshold·calibration을 먼저 확인해야 한다. Bottom-10%는 낮은 coverage의 정확도가 약간 높았지만, 추가 feature의 독립 효과는 matched-coverage·새 validation에서 확인해야 한다.

## 4. 우선순위 3: confidence persistence

한 prefix에서 threshold를 넘었다는 이유만으로 끝내는 대신, 같은 answer가 k개의 연속 prefix에서 높은 confidence를 유지하도록 한다.

$$
\text{stop at }t\text{ if }t\ge t_{\min},\quad
\forall s\in\{t-k+1,\ldots,t\}:\widehat a_s=\widehat a_t,\quad
\widetilde p_s(\widehat a_t)\ge\gamma.
$$

OTHER 조건도 각 prefix에 적용한다. Shared prefix이므로 k회의 독립적인 통계 검정으로 해석하지 않는다.

| min_samples | patience k | cap | Accuracy | Mean tokens | Recoverable errors |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 32 | 81.6% | 13,198.8 | 8 |
| 2 | 2 | 32 | 82.4% | 14,164.9 | 4 |
| 2 | 3 | 32 | 82.4% | 15,186.1 | 3 |

k=2는 이 replay에서 warm-up 이후 남은 조기 오류를 줄였다. k=3은 같은 accuracy 점 추정에서 더 비쌌다. 구조를 늘리지 않고 검증하기 쉬운 candidate다. Cap과 persistence의 조합 효과는 별도 calibration으로 확인한다.

## 5. 우선순위 4: reasoning grouping의 semantic value 검증

현재 PGM과 size-matched random control의 observed-answer winner agreement는 t=8/16/32에서 98.8/99.4/99.4%였다. Probability total variation도 평균 약 0.0032/0.0045/0.0043이었다. Seed 43/44에서도 두 방법의 NLL은 비슷했다.

단, gate가 완전히 collapse한 것은 아니다. Singleton partition의 평균 weight는 t=2에서 79.5%, t=8에서 55.3%, t=32에서 21.4%였다. Late prefix에서는 all-one과 intermediate partition에도 weight가 간다. 그러나 t=32에 partition별 observed-answer winner가 다른 질문은 4%였다. 같은 support head를 고정했을 때 gate만 바꿔 answer selection을 달리할 기회가 작다는 뜻이다. Probability calibration 효과까지 없다는 뜻은 아니다.

현재 cosine 평균은 same-answer pair 0.9849, different-answer pair 0.9766이며 분포가 크게 겹친다. 높은 cosine 자체는 shared failure pattern의 검증이 아니다. 기존 middle-layer·early-80%·centering 실험도 확정적인 추가 효과를 보이지 않았으므로 같은 pooling 후보를 단순히 반복하는 것은 우선순위가 낮다.

후속 representation 실험은 train에서 100–200 path pair를 확인해 same method, same wrong assumption, wording similarity를 구분하는 것부터 시작한다. 그 후 equation/step representation이나 question-conditioned relation score를 비교한다. Same wrong assumption은 단순한 same-answer 또는 both-wrong label과 동일하지 않다.

Semantic entropy 연구는 meaning과 wording의 구분을 강조하지만, answer entailment cluster는 reasoning/error-pattern cluster와 목적이 다르다. 해당 논문도 consistent errors는 semantic entropy가 다루기 어려운 경우로 설명한다. 그대로 교체하면 shared-error grouping이 해결된다고 가정하지 않는다. [Farquhar et al., Nature 2024](https://www.nature.com/articles/s41586-024-07421-0).

추가 control로 total evidence mass를 맞춘 weighting을 비교할 수 있다. 현재 cluster 수는 answer별 상대 evidence와 전체 score scale을 함께 바꾼다. Equal-total-weight control은 semantic reweighting과 confidence-scale 효과를 분리하는 진단용이다. 이런 control이 원래 discount의 장점을 일부 제거할 수 있으므로 새 기본식으로 바로 채택하지 않는다.

## 6. Partition disagreement를 stopping feature로 사용하기

Gate entropy만 높아도 모든 partition이 같은 answer distribution을 내면 answer uncertainty는 증가하지 않는다. 다음 값은 fitted conditional mixture에서 partition에 따른 predictive disagreement를 나타낸다.

$$
U_Z(D_t)=H\!\left(\sum_m\pi_m p_m\right)-\sum_m\pi_m H(p_m).
$$

고정한 t=16에서 이 값의 평균은 correct 0.0296, incorrect 0.0807이었고, t=32에서는 0.0426/0.1698이었다. 하지만 t=4에서는 correct/incorrect가 0.00439/0.00407로 분리되지 않았다. 따라서 모든 prefix에 같은 disagreement gate를 넣는 근거는 약하다.

후속 calibration head에는 t, top-answer probability, OTHER probability, U_Z를 함께 넣고, U_Z의 추가 효과를 question-level validation으로 확인한다. Random-partition U_Z와 동일 feature의 non-PGM calibrator를 반드시 비교한다. 모든 partition이 같은 wrong answer에 동의하면 U_Z로 해결되지 않는다.

## 7. Support quality와 grounding

Original policy의 wrong 103문제는 다음과 같이 나뉜다.

| 상황 | 문제 수 |
|---|---:|
| Correct answer가 stop 시점에 이미 있었음 | 41 |
| Stop 이후에 등장했지만 32개 bank 안에는 있었음 | 26 |
| 32개 bank 어디에도 correct answer가 없었음 | 36 |

41문제는 추가 sampling 이전에 answer selection/quality 추정의 여지가 있었다. 현재 support MLP는 probability·entropy·length features를 받으며, question과 answer가 실제로 수학적으로 맞는지 직접 검사하지 않는다. Softplus support는 모두 양수다.

후속 후보는 train-only response correctness auxiliary loss와 question-grounded feature다. `4^x=8`에서 candidate x=8은 substitution으로 반박할 수 있다. 이런 constraint check가 적용되지 않는 질문은 missing으로 처리하고 비용과 coverage를 보고한다. Verifier를 추가하면 같은 verifier를 non-PGM 비교군에도 제공해 추가 정보의 이득과 grouping의 이득을 분리한다. Same-source confidence가 높은 systematic error에는 더 복잡한 latent graph만으로 추가 정보가 생기지 않는다.

## 8. Calibration policy 선택의 통계적 안정성

현재 150-question calibration-B에서 1%p tolerance는 약 1–2문제 차이에 해당한다. 여러 threshold를 탐색해 얻은 empirical constraint는 test의 accuracy guarantee가 아니다.

예를 들어 harmful-disagreement event가 150개에서 한 번도 관측되지 않아도, 그 event probability의 one-sided exact-binomial 95% upper bound는 약 1.98%다. 이 보수적 event-risk bound를 1% 아래로 내리려면 zero event 기준 최소 299개의 독립적인 question이 필요하다. 이는 benefit을 포함한 signed accuracy difference에 대한 직접적인 proof가 아니다.

후속에서는 대표성 있는 더 큰 calibration set과 predefined policy family를 사용하고, policy-level risk 검증을 분리한다. Learn then Test처럼 finite policy family를 risk test로 다루는 방법을 검토할 수 있으나, current calibration/test shift와 question 내 prefix dependence를 무시한 채 guarantee를 옮겨 쓰지 않는다. [Angelopoulos et al., Learn then Test](https://arxiv.org/abs/2110.01052).

## 9. Serving과 implementation

현재 online timing은 question batch 8의 한 group이 모두 종료돼야 다음 group을 시작한다. 한 질문이 32개까지 가면 먼저 끝난 질문의 slot은 비어 있다. Work-conserving queue로 종료 질문을 다음 질문으로 대체하면 adaptive methods의 batch utilization을 개선할 수 있다. 같은 scheduler를 CGES와 PGM 모두에 적용해야 한다. 현재 timing 결과는 공통 round/group 조건의 측정으로 보고한다.

현재 online PGM은 새 response마다 전체 history의 prefix들을 다시 구성한다. OnlinePartitions state, per-cluster/per-answer support sum, confidence summary를 cache하고 한 번만 update하는 구현으로 바꿀 수 있다. Cluster size가 변하면 그 cluster의 이전 contribution을 빼고 새 contribution을 넣는다. Prefix별 probability가 기존 구현과 일치하는지 검증한 뒤 latency를 측정한다.

Generation 단계에서 pooled hidden state와 필요한 raw statistics를 함께 수집하면 별도 TF 비용을 줄일 가능성이 있지만 backend/model별 구현과 alignment 검증이 필요하다. Sampled-token logprob만으로 entropy나 full-vocabulary self-certainty를 복원할 수는 없다. 우선순위는 generation budget 감소와 queue utilization이며, feature pass 제거는 이후의 engineering 실험이다.

## 10. 권장 후속 순서

1. Frozen heads로 minimum·patience·cap·gamma를 calibration에서 공동 선택하고, 새 holdout에서 검증한다. 현재 `min=4, cap=16`은 유망한 후보이지 새 primary winner가 아니다.
2. Early-prefix bucket loss와 prefix-aware OTHER calibration을 비교한다. Architecture를 바꾸지 않는 control을 포함한다.
3. Train pair annotation으로 representation의 semantic value를 검증한 뒤 support quality/representation을 바꾼다. Random grouping과 같은 정보의 non-PGM 모델을 유지한다.
4. Train bank의 future-prefix supervision으로 continuation-value head를 학습한다. Corrected/regressed outcome과 token cost를 모두 반영한다.
5. Work-conserving scheduling과 incremental inference를 동일 정책에서 검증한다. 정책 개선과 serving 개선을 분리해 보고한다.

Adaptive alpha·더 깊은 gate·더 많은 partition은 위 단계에서 relation signal과 conditional predictive disagreement의 추가 가치가 확인된 뒤 검토한다. 현재 frozen-head alpha sweep은 alpha를 바꿔도 fixed-budget accuracy@32가 대부분 83.0%였고 probability/NLL이 더 크게 변했다. 이는 재학습 ablation이 아니므로 최적 alpha를 선택하는 결과로 쓰지 않는다.

## 11. PGM-specific 확장 후보의 조건

Generic warm-up·patience·cap 개선은 모든 adaptive 비교군에도 적용한다. 그 뒤에도 남는 추가 이득이 PGM-specific 기여다.

현재 gate는 geometry summary만 받고 각 partition에 들어 있는 response quality를 직접 보지 않는다. Confidence/answer-conflict summary를 더한 quality-aware gate를 비교할 수 있다. 다만 현재 partition별 winner가 대부분 같으므로 먼저 representation의 추가 정보를 확인해야 한다.

Cluster reliability gate를 추가하는 경우 다음 score family를 시험할 수 있다.

$$
S_m(y)=\sum_k r_{m,k}\,n_{m,k}^{\alpha-1}\sum_{i\in k}s_i\mathbf1[a_i=y],\qquad r_{m,k}\in[0,1].
$$

r은 cluster quality feature로 학습하는 deterministic gate다. 이를 latent reliability variable에 대한 exact marginalization과 동일하다고 표현하지 않는다. 같은 quality gate를 사용하는 single-partition·non-PGM 비교군을 포함한다. 좋은 quality feature 없이 reliability 변수만 추가하는 것은 현재 관측 결과로 뒷받침되지 않는다.

독립 holdout에서 symbolic answer equivalence와 answer extraction도 audit한다. 기존 로그의 comparison timeout 경고만으로 grading error가 있다고 단정하지 않는다. Grader를 수정하게 되면 모든 방법을 같은 새 grader version으로 다시 평가하고 원래 결과와 변경 이유를 보존한다.

---

## Historical source: `PGM_ASC_revised_method_and_experiments.md`

# PGM을 이용한 Adaptive Self-Consistency: 수정 방법 및 실험 계획

**제안 제목:** Probabilistic Aggregation of Reasoning Patterns for Adaptive Self-Consistency
**작성일:** 2026-10-29
**기반 문서:** [Modeling Reasoning Dependencies for Adaptive Self-Consistency](references/research_proposal.pdf)
**상태:** 구현 전 연구 설계. 아래 가설, 설계 선택, 성능 목표는 실험으로 검증해야 하며 관측된 결과가 아니다.

## 1. 연구 목표와 범위

답 빈도와 개별 응답 confidence에 더해 추론 경로 사이의 관계를 사용하면, 같은 정확도에서 adaptive self-consistency의 전체 추론 비용을 줄일 수 있는지 검증한다. 특히 추론 패턴을 하나로 확정하는 대신 여러 가능한 패턴 할당을 확률적으로 집계하는 것이 실제 답 선택과 종료 결정을 개선하는지 확인한다.

연구의 핵심 질문은 다음과 같다.

> 추론 패턴 구분의 불확실성을 주변화하면, 답 빈도·confidence·단순 중복 제거보다 더 정확하고 저렴하게 sampling 종료를 결정할 수 있는가?

### 1.1 검증할 가설

- **H1 — 관계 정보의 추가 가치:** 답 빈도와 confidence를 통제한 뒤에도 추론 관계 특징이 정답 여부에 대한 추가 예측력을 제공한다.
- **H2 — 패턴 불확실성의 가치:** 단일 partition을 사용하는 방법보다 여러 partition을 주변화하는 방법이 종료 시점의 과신과 회복 가능한 조기 종료 오류를 줄인다.
- **H3 — 전체 비용의 개선:** 관계 특징 계산과 PGM 추론을 포함해도 정확도–비용 곡선이 개선된다.

H1을 먼저 검증한다. 관계 정보가 추가 정보를 제공하지 않으면, 복잡한 잠재변수 모형을 도입하기 전에 표현과 문제 설정을 수정한다.

### 1.2 해석상의 범위

고정된 질문, 프롬프트, 모델, decoding 설정에서 응답은 서로 독립적인 난수로 생성한다. 이전 응답을 다음 생성의 프롬프트에 추가하지 않는다.

PGM은 decoder 출력 사이에 실제 생성 의존성이 있다고 가정하는 모형이 아니다. 관측된 추론 특징을 통해 정답에 대한 지지 증거를 집계하는 **조건부 확률모형**이다. 잠재 패턴도 별도 검증 없이 실제 오류 원인이나 인과적 메커니즘으로 해석하지 않는다.

초기 범위는 다음과 같다.

- 고정 프롬프트를 사용하는 adaptive sampling.
- 작은 두 모델에서 수학과 과학 객관식 평가.
- 고정한 LLM의 출력에서 경량 집계기만 학습.
- 텍스트 답의 동치성을 정의하기 어려운 코드 생성은 후속 확장.
- 프롬프트 변경, verifier 호출, 다양성 유도 생성의 선택은 후속 연구.

## 2. 원래 방법에서 수정하는 부분

| 항목 | 원래 계획 | 수정 계획과 검증 목적 |
|---|---|---|
| 같은 패턴의 반복 | 패턴 내부 평균으로 집계 | 반복을 부분적으로 할인하는 강도를 학습하고 평균·합산과 비교 |
| 잠재 패턴 추론 | 전체 할당을 주변화하는 수식 | 소수의 후보 partition에 대한 조건부 mixture로 1차 구현 |
| 잠재변수 해석 | 공유 reasoning/error 패턴 | 증거 집계를 위한 잠재 그룹으로 정의하고 오류 의미는 별도 검증 |
| 확률 보정 | held-out threshold 선택 | 확률 보정과 전체 순차 종료 정책 검증을 명시적으로 분리 |
| 비교군 | 기존 adaptive SC 중심 | 동일한 라벨·특징을 쓰는 학습형 비-PGM 집계기 추가 |
| 평가 | 최종 정확도와 비용 | 답 선택 효과·종료 효과·회복 가능한 조기 오류를 분리 |
| 과제 | 수학·과학·코드 | 수학을 주 개발 과제로 두고 과학 객관식과 모델 간 전이 평가로 확장 |

## 3. 제안 방법

### 3.1 표기와 관측 특징

| 기호 | 의미 |
|---|---|
| $q$ | 질문 |
| $r_i$ | $i$번째 생성 응답의 추론 경로 |
| $a_i$ | 응답에서 추출하고 정규화한 최종 답 |
| $c_i$ | token probability에서 계산한 confidence 특징 |
| $h_i$ | 추론 구간의 pooled hidden-state 표현 |
| $e_i$ | 유사도 계산에 사용하는 정규화된 추론 표현 |
| $x_i$ | 개별 응답의 confidence·길이·표현 특징 |
| $D_t$ | $t$개 응답까지의 관측 history |
| $C_t$ | 현재까지 관측된 정규화 답들의 집합 |
| $\mathcal Y_t$ | 확률분포를 정의하는 후보 답 공간 |
| $z=(z_1,\ldots,z_t)$ | 응답들의 패턴 할당, 즉 partition |
| $n_k$ | partition에서 패턴 $k$에 속하는 응답 수 |
| $Y$ | 정답을 나타내는 예측 대상 |
| $y_q^*$ | 학습·평가용 실제 정답 |
| $N$ | 최대 생성 응답 수 |
| $\gamma$ | 보정 확률에 적용하는 종료 임계값 |
| $\tau$ | 실제 종료 표본 수 |

관측 history는 다음과 같이 둔다.

$$
D_t=\{(r_i,a_i,c_i,h_i)\}_{i=1}^{t}.
$$

초기 confidence 특징은 다음 후보에서 개발 세트로 선택한다.

- 추론 구간의 평균 token log-probability.
- 최종 답 구간의 평균 token log-probability.
- 추론 구간의 평균 token entropy.
- 추론 길이, 답 길이, 토큰 제한에 의한 truncation 여부.

confidence 계산에 원모델 분포와 sampling 분포 중 무엇을 사용하는지 구현 전에 고정한다. 기본안은 temperature 변환 전 원모델 softmax의 log-probability이며, 비교군에도 같은 정의를 적용한다. confidence 특징의 정규화 통계는 학습 데이터에서만 계산한다.

첫 hidden-state 표현은 지정한 Transformer layer의 추론 토큰 평균으로 한다. layer 선택은 개발 세트에서 고정한다. 유사도 계산에는 최종 답 구간과 정형화된 마무리 문구를 제외한 표현을 사용한다.

추론 본문에도 결론 정보가 남을 수 있으므로 이것만으로 답 정보가 완전히 제거되었다고 주장하지 않는다. 결론 문장 마스킹과 표현 종류에 따른 민감도 분석을 수행한다.

토큰별 hidden state와 전체 vocabulary logits를 모두 저장하지 않고, 가능하면 생성 중 합계·평균·entropy를 누적해 응답별 요약만 저장한다. 특징을 얻기 위한 추가 forward pass가 필요하면 그 비용을 포함한다.

### 3.2 온라인 후보 partition 생성

첫 구현에서는 모든 가능한 partition을 탐색하지 않는다. 서로 다른 유사도 임계값을 사용한 온라인 clustering으로 4–8개 후보를 만든다.

각 후보 임계값 $\eta_m$에 대해 다음 절차를 사용한다.

1. 새 응답의 표현 $e_t$와 기존 cluster centroid의 cosine similarity를 계산한다.
2. 최대 유사도가 $\eta_m$ 이상이면 해당 cluster에 넣는다.
3. 그렇지 않으면 새로운 cluster를 만든다.
4. 동률이면 먼저 생성된 cluster를 선택한다.
5. 초기 구현에서는 과거 응답을 다른 cluster로 재할당하지 않는다.

임계값 후보는 학습 데이터의 질문 내부 유사도 분포에서 정하고, 후보 수와 범위는 개발 세트로 고정한다. cluster ID는 첫 등장 순서로 정규화하고, 동일한 partition이 중복되면 하나로 합친다.

현재 prefix에서 가능한 후보 집합을 다음과 같이 표기한다.

$$
\mathcal Z_t=\{z^{(1)},\ldots,z^{(M_t)}\}.
$$

partition은 반드시 $D_t$만으로 만든다. 전체 $N$개 응답을 먼저 clustering한 뒤 초기 prefix에 그 결과를 적용하는 방식은 사용하지 않는다.

이 설계에서 잠재변수는 제한된 후보 partition 중 어떤 할당이 적절한지를 나타낸다. 전체 partition 공간에 대한 불확실성을 완전히 표현하는 것은 아니다.

### 3.3 partition 확률과 조건부 PGM

partition별로 다음과 같은 관계 요약 특징 $v_m$을 계산한다.

- cluster 수와 cluster 크기 분포.
- cluster 내부 유사도의 평균과 분산.
- cluster centroid 사이의 유사도.
- singleton 비율과 할당의 집중도.

partition 생성과 이 gating 특징에는 최종 답 문자열이나 정답 라벨을 입력하지 않는다. gating 모형 자체는 학습 질문의 정답 라벨로 학습할 수 있다.

공유 MLP를 이용해 후보 partition의 확률을 계산한다.

$$
\pi_{t,m}
=p_\phi(z^{(m)}\mid r_{1:t},h_{1:t})
=\frac{\exp f_\phi(v_m)}{\sum_{j=1}^{M_t}\exp f_\phi(v_j)}.
$$

조건부 joint distribution은 다음과 같다.

$$
p_{\theta,\phi}(Y=y,Z=z^{(m)}\mid D_t)
=\pi_{t,m}\,p_\theta(y\mid D_t,z^{(m)}).
$$

여기서 $p_\phi$는 후보 partition에 대한 학습된 조건부 분포이다. 실제 오류 원인에 대한 참 posterior로 간주하지 않는다.

### 3.4 반복 증거의 부분 할인

원래 계획의 패턴별 평균과 응답별 합산을 연결하는 집계식을 사용한다.

$$
\bar g_{\theta,k}(y)
=\frac{1}{n_k}\sum_{i:z_i=k}g_\theta(y,x_i,a_i),
$$

$$
S_{\theta,\alpha}(y;D_t,z)
=b_\theta(y,D_t)
+\sum_{k:n_k>0}n_k^\alpha\bar g_{\theta,k}(y),
\qquad 0\leq\alpha\leq1.
$$

- $\alpha=0$: 패턴당 평균. 원래 계획의 집계 규칙에 해당한다.
- $\alpha=1$: 응답별 지지 점수 합산. partition의 영향이 사라진다.
- $0<\alpha<1$: 같은 패턴의 반복을 부분적으로 할인한다.

처음에는 전역 $\alpha$ 하나만 학습한다. 패턴별 할인 강도는 충분한 데이터와 기본 실험 결과를 확보한 뒤 확장한다.

첫 지지 점수 구현은 다음과 같이 단순하게 제한한다.

$$
g_\theta(y,x_i,a_i)
=s_\theta(x_i)\,\mathbf1[a_i=y],
\qquad
s_\theta(x_i)=\operatorname{softplus}(\operatorname{MLP}_\theta(x_i)).
$$

즉, 개별 응답은 자신이 제시한 답을 양의 점수로 지지한다. 이 선택은 PGM의 효과와 복잡한 verifier의 효과가 섞이는 것을 줄이기 위한 초기 구현이다. raw reasoning을 추가 LLM으로 평가하는 방식은 기본 실험에 포함하지 않는다.

기본 실험에서는 일반 후보 답의 $b_\theta$를 0으로 둔다. $\alpha$ ablation에서 빈도 정보를 별도 bias로 우회 입력하지 않는다. $\alpha=1$에서 partition이 결과에 영향을 주지 않는 성질도 유지한다.

이 할인식은 경험적으로 검증할 집계 함수족이다. 실제 응답 의존성에서 유도된 유효 표본 수 공식이나 Bayesian likelihood라고 주장하지 않는다.

### 3.5 후보 답 공간과 OTHER

객관식 문제는 모든 선택지를 처음부터 후보에 포함한다.

$$
\mathcal Y_t=\{A,B,C,D\}
$$

선택지 개수가 다르면 해당 문제의 전체 선택지를 사용한다. 응답에서 아직 등장하지 않은 선택지도 확률분포에 포함한다.

수학 등 열린 답 공간에서는 다음과 같이 둔다.

$$
\mathcal Y_t=C_t\cup\{\mathrm{OTHER}\}.
$$

정답이 아직 $C_t$에 없으면 해당 학습 prefix의 라벨을 OTHER로 지정한다. OTHER는 여러 미관측 답의 확률을 묶은 상태이며, 그 자체로 미관측 정답의 내용을 생성하지는 않는다.

초기 OTHER logit은 별도 경량 head로 계산한다.

$$
S(\mathrm{OTHER};D_t,z)=u_\psi(w_t).
$$

$w_t$에는 현재 표본 수, 서로 다른 답 수, 답 빈도 분포, confidence 요약, truncation 비율을 사용한다. 이 head는 partition이나 관계 특징을 사용하지 않으며 모든 해당 비교군에서 같은 입력과 학습 절차를 사용한다.

새 답이 관측되면 후보 공간과 조건부 분포를 다시 계산한다. 이것은 동적인 후보 집합 위의 학습형 조건부 예측이며, 후보 공간이 바뀔 때마다 정확한 생성모형의 Bayes update를 수행한다고 주장하지 않는다.

### 3.6 posterior 계산과 답 선택

각 partition에서 답 분포를 계산한다.

$$
p_\theta(y\mid D_t,z)
=\frac{\exp S_{\theta,\alpha}(y;D_t,z)}
{\sum_{b\in\mathcal Y_t}\exp S_{\theta,\alpha}(b;D_t,z)}.
$$

최종 예측 분포는 partition에 대한 주변화로 얻는다.

$$
p_t(y)
=\sum_{m=1}^{M_t}\pi_{t,m}
\,p_\theta(y\mid D_t,z^{(m)}).
$$

계산은 log-sum-exp를 이용해 안정적으로 수행한다. 4–8개 후보 partition 안에서는 합을 직접 계산하며, 전체 partition 공간에 대해서는 근사임을 명시한다.

객관식의 답 선택은 전체 선택지 중 최대 확률 후보를 사용한다. 열린 답 공간에서는 OTHER를 제외한 관측 후보 중 최대 확률 답을 사용한다.

$$
\hat a_t=\arg\max_{a\in\mathcal A_t}p_t(a),
$$

여기서 $\mathcal A_t$는 객관식의 전체 선택지 또는 열린 답 공간의 $C_t$이다. 동률 처리와 답 추출 실패 처리는 모든 방법에서 동일하게 고정한다.

### 3.7 학습

LLM과 초기 추론 encoder는 고정한다. 지지 점수 head, partition gating head, OTHER head, 할인 강도만 학습한다. 고정 encoder와 deterministic partition 생성 덕분에 제한된 mixture의 log-loss를 직접 최적화할 수 있다.

질문별 prefix 집합을 $T_q$라고 하면 다음 손실을 사용한다.

$$
\mathcal L
=\frac{1}{|Q_{\mathrm{train}}|}
\sum_{q\in Q_{\mathrm{train}}}
\frac{1}{|T_q|}\sum_{t\in T_q}
-\log p_t(y_{q,t}^*).
$$

열린 답 공간에서 $y_q^*\notin C_t$이면 $y_{q,t}^*=\mathrm{OTHER}$이다. 질문마다 같은 수의 prefix를 사용하거나 위와 같이 정규화해 긴 trajectory bank를 가진 질문이 과도한 비중을 갖지 않도록 한다.

초기 prefix도 충분히 포함한다. 큰 $t$까지 사용하는 정책은 그 범위의 prefix를 학습·보정에서 다뤄야 하며, $t\leq16$만 학습하고 $t\leq32$로 확장한 결과를 별도 검증 없이 동일 조건으로 취급하지 않는다.

잠재변수 collapse를 확인하기 위해 gating entropy, 선택된 cluster 수, 모든 경로를 하나로 묶거나 전부 분리하는 비율을 기록한다. uniform mixture와 단일 partition도 비교한다.

### 3.8 확률 보정과 종료

데이터는 질문 단위로 train / development / calibration / test로 분리한다. calibration 세트도 확률 보정용 A와 종료 정책 선택용 B로 나눈다.

1. train에서 집계 모형을 학습한다.
2. development에서 특징, 구조, partition 후보, 학습 hyperparameter를 고정한다.
3. calibration-A에서 temperature 등 확률 보정 변수를 학습한다.
4. calibration-B에서 순차 sampling 전체를 재생해 종료 임계값을 선택한다.
5. 모든 설정을 고정하고 test를 한 번의 최종 평가 대상으로 사용한다.

temperature scaling의 예시는 다음과 같다.

$$
\tilde p_t(y)
=\operatorname{softmax}_y\!\left(\frac{\log p_t(y)}{T}\right),
\qquad T>0.
$$

종료 시점은 다음과 같다.

$$
\tau
=\min\left\{
\inf\{t\geq1:\tilde p_t(\hat a_t)\geq\gamma\},\,N
\right\}.
$$

OTHER가 가장 높은 상태에서는 OTHER 자체를 정답으로 반환하거나 그 확률로 종료하지 않는다. 최대 예산에 도달하면 정해진 답 선택 규칙으로 답을 반환하고 예산 종료로 기록한다. 추출 가능한 후보가 전혀 없으면 공통 fallback 규칙을 적용하고 정확도 및 추출 실패율에 포함한다.

임계값은 개별 prefix의 ECE만으로 결정하지 않는다. calibration-B에서 실제 종료 시점의 정확도, 비용, 오류 유형을 함께 측정해 선택한다. 학습된 근사 분포에서 $\gamma=0.95$라는 이유만으로 실제 오답률이 5% 이하라고 주장하지 않는다.

### 3.9 추론 절차

```text
입력: 질문 q, 고정 LLM, 최대 예산 N, 학습된 집계기, 보정 변수, 임계값 gamma
초기화: history, 후보 답 집합, 각 임계값의 온라인 cluster

for t = 1, ..., N:
    고정 프롬프트에서 새로운 응답을 독립적으로 생성한다.
    답, confidence, 추론 표현, 생성 비용을 추출한다.
    현재 history만으로 각 온라인 partition을 갱신한다.
    중복 partition을 합친다.
    partition 확률과 할인된 후보 답 점수를 계산한다.
    partition을 주변화하고 확률을 보정한다.
    반환 가능한 후보 중 최상위 답을 고른다.

    if t < N and 해당 답의 보정 확률 >= gamma:
        조기 종료 사유와 함께 답을 반환한다.

최대 예산 종료 사유와 함께 최종 선택 답을 반환한다.
```

## 4. 실험 계획

### 4.1 1단계: 관계 정보의 추가 가치 검증

최종 평가 세트와 겹치지 않는 개발 문제 약 150개에서 시작한다. 모델 두 개를 사용할 경우 질문당 16개 응답으로 총 4,800개 응답을 생성한다. pilot의 결과를 보고 최종 평가 문제나 라벨을 선택하지 않는다.

다음 예측기를 비교한다.

| 예측기 | 입력 | 검증 목적 |
|---|---|---|
| Count | 표본 수, 답 빈도, 1·2위 차이, 답 entropy | 빈도 정보의 기준 성능 |
| Count + confidence | 위 특징과 개별 confidence 요약 | 개별 응답 신뢰도의 추가 가치 |
| Count + confidence + relation | 위 특징과 유사도·cluster 요약 | H1: 관계 정보의 추가 가치 |
| 개별 응답 표현 집계 | 같은 confidence와 응답 표현 | 표현 자체의 가치와 명시적 관계의 가치 구분 |

질문 단위 분할로 검증하고, 정확도뿐 아니라 NLL과 Brier score를 확인한다. 비슷한 답 빈도와 confidence를 가진 history를 대상으로 한 조건부 분석도 수행한다. matching만으로 인과 효과를 주장하지는 않는다.

다음 네 유형을 함께 분석한다.

| 선택 답의 상태 | 추론 패턴이 비슷함 | 추론 패턴이 다양함 |
|---|---|---|
| 정답 | 올바른 반복 풀이를 과도하게 할인하는가? | 다양한 올바른 지지를 활용하는가? |
| 오답 | 반복 오류의 과신을 줄이는가? | 다양한 오류를 정답 증거로 오인하는가? |

가능하면 100–200개 경로 쌍을 정답 여부와 유사도 구간별로 표집해 사람이 확인한다. 같은 풀이 방법인지, 같은 잘못된 가정을 공유하는지, 문체만 비슷한지를 구분한다. 일부를 두 명이 중복 검토해 일치도를 기록한다. 이 검토는 기본적으로 분석용이며, annotation을 학습에 쓰는 실험은 별도로 구분한다.

**진행 기준:** 관계 특징의 추가 가치가 없으면 본 실험 규모를 늘리기 전에 표현과 grouping을 수정한다. 효과가 불확실하면 pilot의 표본 수와 신뢰구간을 먼저 검토한다.

### 4.2 모델

초기 모델은 다음 두 개로 한다.

- Gemma-3-4B-IT.
- Qwen3.5-4B. thinking mode를 사용할 경우 해당 모드로 고정하고 thinking token을 비용에 포함한다.

작은 모델에서 H1–H3을 검증한 뒤 Gemma-3-12B-IT와 Qwen3.5-9B로 확장한다. 모델 간 비교에서는 규모 차이 자체를 방법의 이득으로 해석하지 않고, 각 모델 내부에서 비교군 대비 차이를 평가한다.

모델 revision, tokenizer, chat template, precision, generation backend, decoding 설정을 기록한다. 원본 계획의 모델 선택은 유지하되 네 모델 전체를 첫 구현의 완료 조건으로 삼지 않는다.

### 4.3 데이터와 분할

| 역할 | 기본안 | 분할 및 해석 원칙 |
|---|---|---|
| 주 학습·개발·보정 | MATH 학습 문제에서 별도 분할 | 모든 prefix와 seed를 질문 단위로 묶어 분리 |
| 주 최종 평가 | MATH-500 | 구조·threshold·모델 선택에 사용하지 않음 |
| 외부 과제 평가 | GPQA Diamond | 기본안은 수학에서 학습·보정한 정책의 과제 간 전이 평가 |
| 과학 in-domain 확장 | Diamond와 중복 없는 별도 과학 객관식 개발 데이터 | 확보 후 별도 학습·보정 프로토콜로 보고 |
| 후속 확장 | MBPP | 프로그램 동치성과 선택·평가용 테스트 분리 후 추가 |

학습·개발·보정 문항 수는 pilot에서 측정한 비용에 따라 고정한다. 정답 라벨을 사용하는 집계기의 학습 규모를 반드시 보고한다. 과학 개발 데이터가 없는 상태에서 GPQA 성능을 in-domain calibrated 성능으로 표현하지 않는다.

GPQA Diamond는 198문항이므로 약 1%p 차이가 두 문제 정도의 차이에 해당한다. 작은 차이는 질문 단위 신뢰구간과 실제 달라진 문제 수를 함께 제시하고, 이 세트만으로 미세한 우월성을 단정하지 않는다.

모델 간 전이 실험은 다음 두 조건을 구분한다.

- **완전 전이:** 집계기, 정규화 통계, 확률 보정, threshold를 모두 고정한다.
- **보정 허용 전이:** 집계기는 고정하되 대상 모델의 별도 보정 데이터로 정규화·보정·threshold를 다시 정한다.

두 조건을 같은 무학습 전이 결과로 묶지 않는다.

### 4.4 sampling과 답 처리

- pilot은 $N=16$, 주 실험은 $N=32$를 기본안으로 한다.
- 같은 모델의 모든 방법에 프롬프트, 생성 모드, temperature, top-p, 응답별 토큰 제한을 동일하게 적용한다.
- 각 응답은 이전 응답이 없는 고정 프롬프트에서 독립적으로 생성한다.
- 출력 token limit으로 잘린 응답과 답 추출 실패를 삭제하지 않는다.
- 수학은 답 정규화와 수식 동치성 판정 규칙을 고정한다.
- 객관식은 선택지 정규화, 답 추출, 동률 처리를 고정한다.
- 여러 응답 순서를 비교하되 질문 단위의 독립 표본 수가 늘어난 것으로 취급하지 않는다.

### 4.5 비교군

| 범주 | 비교군 | 필요성 |
|---|---|---|
| 고정 예산 | 단일 응답, SC@8/16/32 | 원래 LLM 성능과 test-time scaling 기준 |
| 빈도 기반 adaptive | ASC, ESC | 답 빈도에 의한 종료와 비교 |
| confidence 기반 adaptive | ReASC, CGES | 개별 confidence 이상의 이득 확인 |
| 단순 관계 집계 | 클러스터당 한 표, 패턴 평균, 중복 제거 후 confidence 합산 | 복잡한 확률모형의 필요성 확인 |
| 지도학습 집계 | 빈도·confidence 기반 logistic regression 또는 MLP | 추가 라벨의 효과 통제 |
| 동일 표현 집계 | confidence와 응답 표현을 입력받는 DeepSets | 응답 표현과 지도학습의 효과 통제 |
| 가까운 선행 방법 | RASC, VecCISC 또는 CHSC-gated ESC | reasoning 활용·중복 제거·false-consensus gate와 비교 |

최소 본 실험에는 SC, ASC, ESC, ReASC, CGES, 단순 cluster 집계, 지도학습 비-PGM 집계기를 포함한다. 가까운 선행 방법 중 추론 유사도와 hidden state를 쓰는 비교는 논문 수준의 차별성 검증에 우선 추가한다.

같은 라벨과 표현을 사용하는 학습형 비교군에는 같은 train/development/calibration 분할을 적용한다. PGM만 확률을 보정하거나 더 넓은 threshold 탐색을 수행하지 않는다. 가능한 범위에서 학습 용량과 hyperparameter 탐색 예산을 맞춘다.

추가적인 최신 비교 후보는 다음과 같다.

- CASE: confidence 기반 선택과 종료의 강한 비교 후보.
- ASC-D: 객관식에서 답 확률벡터를 얻을 수 있을 때의 추가 비교 후보.

ASC-D 등의 최빈 답 식별 보장은 실제 정답 보장과 구분한다. 외부 critic이나 reward model을 사용하는 방법은 추가 모델의 호출·토큰·시간을 모두 포함하고 입력 정보 차이를 명시한다.

### 4.6 답 선택과 종료 효과의 분리

세 실험을 구분한다.

1. **고정 예산 답 선택:** $t=8,16,32$에서 동일한 응답을 보고 majority vote, confidence 합산, cluster 집계, PGM이 어떤 답을 고르는지 비교한다.
2. **종료 규칙 비교:** 답 선택기를 고정한 상태에서 빈도 기반 종료와 posterior 기반 종료를 비교한다. 동일 답 선택기의 예측이 가진 confidence를 사용하도록 정책을 명시한다.
3. **전체 시스템 비교:** 각 방법의 원래 답 선택과 종료를 결합한 결과를 비교한다.

이렇게 하여 최종 정확도 개선이 더 좋은 답 선택 때문인지, 더 적절한 종료 시점 때문인지 구분한다.

### 4.7 필수 ablation

| 변경 | 검증할 질문 |
|---|---|
| confidence 제외 | 개선이 관계 정보만으로도 남는가? |
| 관계 특징과 grouping 제외 | 개별 응답 특징만으로 설명되는가? |
| 최적 단일 partition | 여러 할당의 주변화가 필요한가? |
| uniform mixture | 학습한 gating이 필요한가? |
| $\alpha=0$ | 원래 패턴 평균이 충분한가? |
| $\alpha=1$ | 중복 할인이 필요한가? |
| 학습한 전역 $\alpha$ | 부분 할인이 두 극단보다 나은가? |
| 무작위 partition | 유사도에 의한 구조 자체가 중요한가? |
| 결론 문장 마스킹 | 관계 특징이 결론 정보에만 의존하는가? |
| hidden-state 표현과 텍스트 표현 비교 | 특정 표현에만 효과가 제한되는가? |

무작위 partition은 가능한 한 cluster 수와 크기 분포를 맞춰 단순한 집계 규모 변화와 의미 있는 grouping의 효과를 구분한다.

## 5. 평가 지표와 통계 분석

### 5.1 주 성과 지표

같은 정확도에서 평균 총비용이 얼마나 줄어드는지를 주 지표로 한다.

$$
\min\mathbb E[\mathrm{Cost}]
\quad\text{subject to}\quad
\mathrm{Acc}_{\mathrm{method}}
\geq\mathrm{Acc}_{\mathrm{SC}@N}-\epsilon.
$$

허용 정확도 감소폭 $\epsilon$은 최종 평가 전에 정한다. 예를 들어 1%p를 목표로 둘 수 있지만, 평가 문항 수가 그 차이를 판별할 만큼 충분한지 함께 검토한다. 평균값이 조건을 만족한다는 사실만으로 통계적 비열등성이 입증되었다고 표현하지 않는다.

다음 두 비교를 함께 제시한다.

- 같은 정확도에서 비용 비교.
- 같은 비용에서 정확도 비교.

SC@32뿐 아니라 가장 강한 adaptive 비교군 대비 결과도 제시한다. test 결과를 보고 유리한 threshold만 선택하지 않으며, 주 operating point는 calibration 데이터에서 고정한다. 사전에 고정한 threshold grid의 전체 곡선은 보조 결과로 제시한다.

### 5.2 비용

다음 항목을 분리해 기록한다.

- 생성 응답 수.
- 생성 token 수: thinking token, 잘린 출력 포함.
- 특징 추출 및 추가 forward pass 비용.
- 임베딩·유사도 계산 시간.
- partition 갱신·PGM 추론 시간.
- 외부 critic 또는 verifier 비용.
- 실제 전체 wall-clock time과 GPU 메모리 사용량.
- 집계기 학습·확률 보정에 들어간 일회성 비용.

추론 비용과 학습·보정 비용은 별도로 보고한다. 필요하면 배포 질문 수에 따라 일회성 비용을 상각한 결과도 제시한다.

기본적인 token 예산은 다음 식으로 사전 산정한다.

$$
B_{\mathrm{gen}}
\approx
\sum_{m\in\mathrm{models}}\sum_{q\in Q_m}
N_q\,\bar L_{m,q}.
$$

pilot에서 평균 응답 길이와 GPU 처리량을 측정한 뒤 학습 문항 수와 확장 모델 수를 정한다. GPU 정보 없이 실행 시간을 임의로 확정하지 않는다.

### 5.3 calibration과 종료 오류

| 지표 | 정의 및 목적 |
|---|---|
| Final accuracy | 실제 반환 답의 정확도 |
| Stopping-time NLL/Brier | 실제 종료 시점의 확률 품질 |
| Selected-answer calibration | 반환 답에 부여한 확률과 실제 정오 여부의 관계 |
| Early-stop rate | $\tau<N$인 질문 비율 |
| Early-stop error | $\tau<N$인 질문에서의 조건부 오답률 |
| Budget-exhaustion rate | 최대 예산까지 진행한 질문 비율 |
| Budget-exhaustion error | 예산 종료 질문의 오답률 |
| Recoverable premature error | 일찍 오답을 반환했지만 같은 선택기가 $N$개 응답에서는 정답을 고르는 비율 |

회복 가능한 조기 종료 오류는 다음과 같이 정의한다.

$$
R_{\mathrm{recover}}
=\Pr\bigl(
\tau<N,\;A(D_\tau)\neq y_q^*,\;A(D_N)=y_q^*
\bigr).
$$

여기서 $A$는 동일하게 고정된 답 선택기이다. 조기 오답이더라도 $N$개까지 생성해 고칠 수 없는 문제와, 추가 생성으로 고칠 수 있었던 문제를 구분한다. 추가 생성이 오히려 정답을 오답으로 바꾸는 경우도 별도로 기록한다.

열린 답 공간의 multiclass NLL/Brier는 정답이 미관측일 때 OTHER로 매핑해 계산한다. 이 값과 실제 반환 답의 정오 여부를 대상으로 한 calibration을 구분한다.

### 5.4 통계 분석

- 방법 간 비교는 동일 질문·동일 응답 bank를 사용한 paired 분석으로 수행한다.
- 신뢰구간은 질문 단위 paired bootstrap을 기본으로 한다.
- seed와 응답 순서 반복은 질문 내부 반복으로 묶는다. prefix·seed·응답 쌍을 독립 질문처럼 취급하지 않는다.
- 주 평가 지표와 operating point를 사전에 고정하고, 많은 비교에서 우연히 유리한 결과만 선택하지 않는다.
- 작은 정확도 차이는 실제 정답이 달라진 질문 수와 함께 보고한다.
- 신뢰구간이 넓으면 승리나 실패를 단정하지 않고 불확실한 결과로 보고한다.

## 6. 응답 bank 재생과 실제 온라인 검증

모든 방법은 같은 최대 $N$개 응답 bank를 공유한다. 각 방법은 자신의 종료 시점까지의 prefix만 볼 수 있다. 정답, 미래 응답, 미래 cluster 정보는 답 선택과 종료에 사용하지 않는다.

재생 평가에서 다음 정보를 저장한다.

```text
question_id, dataset, split, model_id, model_revision, seed
response_index, reasoning_text, extracted_answer, canonical_answer
confidence_features, pooled_representation, generated_token_count
truncated, extraction_failed, generation_time, feature_time
prefix_partition_summaries, partition_weights, alpha
answer_probabilities, selected_answer, stopping_reason
```

평가용 정답은 정책 입력과 분리해 저장한다. 응답 기록에 정답 라벨이 있더라도 inference 함수에는 전달하지 않는다.

재생만으로 실제 serving 시간 절감을 입증할 수는 없다. 재생 결과로 생성 token 비용과 정책 결정을 비교한 뒤, 사전에 정한 test subset에서 실제 온라인 생성을 수행해 전체 시간을 확인한다. GPU, batching, warm-up, feature 추출 방식을 동일하게 통제한다.

기본 구현의 계산량은 각 새 응답을 기존 centroid와 비교하므로 누적 최악의 경우 partition 수 $M$에 대해 $O(Mt^2d)$의 유사도 계산을 가질 수 있다. 여기서 $d$는 표현 차원이다. cluster별·답별 점수 합을 캐시해 불필요한 재계산을 줄이고, 실제 측정값으로 overhead를 보고한다.

## 7. 실패 조건과 후속 판단

| 관측 결과 | 해석 | 다음 행동 |
|---|---|---|
| 관계 특징이 추가 정보를 주지 않음 | 현재 표현이나 grouping이 정답 추정에 부적합할 수 있음 | 문체·결론 정보와 풀이 구조를 구분하는 표현 분석부터 수행 |
| 단순 cluster 집계와 PGM이 같음 | 복잡한 주변화의 실익이 확인되지 않음 | 주장 범위를 단순 집계의 효과로 제한하거나 불확실한 경계 사례에 집중 |
| $\alpha=1$이 가장 좋음 | 해당 조건에서 중복 할인의 필요성이 확인되지 않음 | 할인 가정을 유지하려고 결과를 선택하지 않고 적용 조건을 분석 |
| calibration만 개선되고 비용은 증가 | 신뢰성 개선은 있지만 효율 개선은 입증되지 않음 | 두 결과를 분리해 보고하고 overhead·예산 소진 원인을 분석 |
| 반복 오답을 감지해도 계속 예산만 소진 | 불확실성 감지와 추가 sampling의 가치가 다름 | 후속 연구로 다양성 유도나 verifier 선택을 검토 |
| 다른 과제·모델에서 효과가 사라짐 | 관계 특징 또는 confidence가 분포에 의존 | 완전 전이와 보정 허용 전이를 구분해 적용 범위를 보고 |

기본 연구에서 불확실성이 높다는 이유만으로 추가 sampling이 항상 유익하다고 가정하지 않는다. 동일 분포의 추가 응답으로 교정할 수 없는 체계적 오류가 있을 수 있다.

## 8. 실행 일정

8주 일정은 연구 관리용 기본안이다. 실제 생성 규모는 pilot에서 측정한 자원 사용량으로 확정한다.

| 기간 | 작업 | 완료 산출물 및 의사결정 |
|---|---|---|
| 1주 | 생성·답 정규화·특징 추출 파이프라인, SC/ASC/ESC 재현 | 공통 응답 기록과 재생 절차 |
| 2주 | pilot, 관계 특징의 추가 가치와 네 유형 분석 | H1 검토, 표현 수정 또는 본 구현 진행 결정 |
| 3주 | 지도학습 비-PGM 집계기와 단순 cluster 비교군 | 추가 학습·표현·grouping 효과 분리 |
| 4주 | 후보 partition mixture와 할인 강도 학습 | PGM 기본 구현 및 고정 예산 비교 |
| 5주 | calibration과 순차 종료 정책 검증 | 주 operating point와 평가 규칙 고정 |
| 6주 | 두 모델의 본 평가와 실제 온라인 시간 측정 | 정확도–비용 곡선과 overhead 분석 |
| 7주 | ablation, 과제·모델 전이, 실패 사례 | H2/H3 검토와 적용 범위 분석 |
| 8주 | 질문 단위 통계 분석과 결과 정리 | 논문 초안, 설정 파일, 재현 가능한 결과 표 |

## 9. 연구의 성공 기준

다음 조건을 순서대로 확인한다.

1. 관계 특징이 빈도와 confidence 이상의 추가 정보를 제공한다.
2. 같은 라벨과 특징을 사용하는 비-PGM 집계기보다 PGM의 추가 효과가 확인된다.
3. 패턴 불확실성의 주변화가 단순한 partition 선택보다 실제 결정에 도움이 된다.
4. 보정된 종료 정책이 회복 가능한 조기 오류를 줄이거나, 같은 정확도에서 비용을 절감한다.
5. 그 이득이 전체 추론 overhead를 포함한 결과에서도 유지된다.

모든 조건이 성립하지 않으면 확인된 범위만 기여로 주장한다. 특정 구조를 사용했다는 사실 자체보다, 어떤 조건에서 관계 정보와 확률적 집계가 필요한지 보여주는 것을 연구의 성과로 삼는다.

## 10. 관련 문헌과 차별성

아래 문헌은 원래 계획과 수정안의 비교 대상을 정하기 위한 목록이다. 최신 문헌을 모두 망라한 목록은 아니다.

| 문헌 | 이번 계획과의 관계 |
|---|---|
| [ReASC: Reliability-Aware Adaptive Self-Consistency](https://aclanthology.org/2026.findings-acl.1085/) | confidence와 빈도에 의한 증거 누적 비교군 |
| [CGES: Confidence-Guided Early Stopping](https://arxiv.org/html/2511.02603v2) | confidence 기반 posterior와 종료, 이상적 보장과 근사 점수의 구분 |
| [RASC: Reasoning Aware Self-Consistency](https://aclanthology.org/2025.naacl-long.184/) | reasoning 품질과 답 일관성을 이용하는 조기 종료 선행연구 |
| [VecCISC](https://aclanthology.org/2026.findings-acl.1305/) | 추론 유사도로 중복을 줄여 critic 평가 비용을 낮추는 가까운 연구 |
| [When Agreement Fails: Detecting False Consensus via Hidden-State Consistency](https://link.springer.com/chapter/10.1007/978-981-92-3520-9_27) | hidden-state gate로 잘못된 합의에 의한 종료를 보완하는 가까운 연구 |
| [Diversity Combining for Multi-Path LLM Reasoning](https://arxiv.org/abs/2609.38829) | 경로 정확성의 상관, 유효 표본 수, 예산 선택의 관련 분석 |
| [Adaptive Self-Consistency: From Black-Box Sampling to Distribution-Valued Feedback](https://arxiv.org/abs/2609.38931) | 전체 답 확률벡터를 활용하는 최빈 답 식별과 종료 비교 후보 |
| [CASE: Filter by Quality, Stop by Difficulty](https://aclanthology.org/2026.findings-emnlp.1983/) | confidence를 이용한 선택·종료의 추가 비교 후보 |

차별성 주장은 “reasoning 정보를 처음 사용한다” 또는 “유사도로 중복을 처음 제거한다”가 아니다. **패턴 할당 불확실성의 주변화와 학습한 반복 할인으로, 단순 관계 집계 및 동일 정보의 학습형 비교군보다 순차적 답 선택·종료를 개선하는지**를 검증한다.

## 11. 실행 결과와 후속 실험

2026-10-29에 최초 pilot 150문제 × 16응답을 완료했다. 기존 마지막 layer 평균 표현의 cosine 관계 특징은 사전 지정 t=8에서 추가 효과가 확인되지 않았다. 원래 분석과 응답 bank는 보존한다.

- [최초 pilot 결과 보고서](./artifacts/runs/gemma3_math_pilot_v1/analysis/result_summary.md)
- [vLLM 전환 및 2단계 사전 계획](./STAGE2_EXPERIMENT_PLAN.md)
- [vLLM 구현과 실행 안내](./README.md)

2단계는 vLLM 생성과 batched teacher forcing 특징 추출을 분리한다. 텍스트 개발 136문항에서 중간 layer·초반 80% pooling·prefix 내부 중심화와 정규화 강도를 nested CV로 비교하고, 설정을 고정한 뒤 겹치지 않는 새 150문항에서 검증한다. 원래 PGM 전체의 효과를 입증한 결과와 구분한다.

2단계 실행과 독립 검증도 같은 날 완료했다. 고정한 관계 표현은 마지막 layer의 prefix 내부 중심화, C=0.1이었다. 독립 검증 t=8의 NLL 차이는 -0.00978(95% CI [-0.02460, +0.00302]), Brier 차이는 -0.00424(95% CI [-0.00960, +0.00039])로 개선 방향이었지만 확정적이지 않았다. 종료 replay에서는 관계 모델의 추가 비용 절감이 관측되지 않았다.

- [vLLM 2단계 최종 보고서](./artifacts/runs/gemma3_stage2_vllm_v1/report.md)

2026-10-31에 후보 partition mixture를 직접 구현하는 다음 단계를 구체화했다. 초기 계획의 N=32, OTHER, 학습형 반복 할인, 분리된 확률 보정·종료 정책 선택, MATH-500 최종 평가를 유지한다. 기존에 결과를 본 vLLM 286문제는 구현·학습 자료로 재사용하고 새로운 평가 자료로 취급하지 않는다.

- [PGM 구현 및 본 실험 실행 계획](./STAGE3_PGM_EXPERIMENT_PLAN.md)

---

## Historical source: `STAGE2_EXPERIMENT_PLAN.md`

# vLLM 전환 및 2단계 실험 사전 계획

작성일: 2026-10-29. 이 문서는 2단계 결과를 보기 전에 작성한다. 1차 pilot은 보존하며 이번 실험의 개발 자료로만 사용한다.

## 목표

1. 생성 엔진을 vLLM으로 교체하고, 저장된 token ID에 대한 batched teacher forcing으로 raw confidence·entropy·다층 표현을 추출한다.
2. 1차 pilot의 부정적인 결과가 마지막 layer 평균 표현, 결론 구간, 표현의 공통 방향, 정규화 강도에 의존하는지 검증한다.
3. 표현과 hyperparameter를 개발 문항 안에서 선택하고, 겹치지 않는 독립 문항에서 단 한 번 주 비교를 수행한다.

## 보존 및 공정성

- 기존 Transformers 응답 2,400개와 원래 분석 결과는 수정하지 않는다.
- 개발·검증 응답을 모두 같은 vLLM backend에서 생성한다. 기존 문항의 vLLM 재생성은 이미 본 질문이므로 독립 검증으로 취급하지 않는다.
- 모델 revision, prompt template, BF16, temperature 0.7, top-p 0.95, top-k 해제, 최대 4,096 새 토큰을 유지한다.
- vLLM 환경은 `.venv-vllm`, 특징 추출과 분석 환경은 기존 `.venv`로 분리한다. 기본 홈 캐시 대신 작업 폴더 캐시를 사용한다.
- 생성한 정확한 token ID를 그대로 teacher forcing한다. 문자열 재tokenization으로 응답을 바꾸지 않는다.
- sampled-token raw log-probability를 vLLM과 Transformers에서 대조한다. entropy는 vocabulary 전체를 GPU에서 계산한 뒤 scalar만 저장한다.

## 데이터

- 개발: 기존 pilot 150문항 중 `[asy]` 코드가 없는 136문항.
- 독립 검증: 기존 pilot·smoke와 겹치지 않는 MATH train 문항 150개. `[asy]` 없는 문항에서 난이도별 30개씩 사전 표집한다.
- 점검: 기존 별도 smoke에서 4문제 × 4응답.
- 질문당 16응답, prefix 2/4/8/12/16을 사용한다.
- 도형 코드 제외는 2단계의 모집단 정의다. 기존 1차 결과에서 도형 문제를 사후 삭제하거나 원래 결과를 대체하지 않는다.
- MATH-500과 GPQA Diamond는 이번에도 최종 평가용으로 남긴다. 이번 독립 검증은 개발 연구 안의 holdout이며 최종 benchmark가 아니다.

## 표현 후보

| 후보 | layer | pooling 구간 | prefix 내부 중심화 |
|---|---|---|---|
| final_full | 마지막 normalized layer | 최종 boxed 답 이전 추론 토큰 | 없음 |
| final_early80 | 마지막 normalized layer | 위 구간의 앞 80% | 없음 |
| middle_full | hidden_states[17] | 최종 boxed 답 이전 추론 토큰 | 없음 |
| middle_early80 | hidden_states[17] | 위 구간의 앞 80% | 없음 |
| final_centered | 마지막 normalized layer | 전체 추론 구간 | 있음 |
| middle_centered | hidden_states[17] | 전체 추론 구간 | 있음 |

`early80`은 의미적 오류 검출기가 아니라, 후반 결론 정보에 대한 의존도를 확인하는 고정 구간 마스킹이다. 실제 오류 원인을 복원한다고 주장하지 않는다.

중심화는 현재 관측 prefix의 응답 벡터만으로 계산한다. 각 응답 벡터를 L2 정규화한 뒤 현재 prefix 평균을 빼고 다시 정규화한다. 미래 응답, 질문 간 전체 평균, holdout 통계를 사용하지 않는다.

confidence 특징과 관계 요약 통계는 기존 pilot 정의를 유지한다. 지지 경로의 평균 cosine 등만 바꾸며 최종 답 선택은 다수결로 고정한다.

## 모델 선택

- 예측기 B: 빈도 + confidence.
- 예측기 C: B + 선택한 표현의 관계 특징.
- Logistic regression의 C 후보는 0.001/0.01/0.1/1/10이다.
- 개발 질문에서 outer 5-fold / inner 3-fold nested CV를 수행한다.
- 모든 prefix를 같은 질문 그룹으로 묶는다. 표준화는 각 학습 fold에서만 fit한다.
- inner fold의 t=8 NLL로 baseline의 C와 관계 모델의 표현·C를 각각 선택한다.
- outer test fold는 그 fold의 선택에 사용하지 않는다.
- 개발 문항 전체의 질문 단위 CV로 최종 표현·C를 선택한 뒤 `selection_lock.json`에 기록한다.
- 독립 검증의 라벨이나 지표를 사용해 위 선택을 수정하지 않는다. 평가 전에 lock hash를 확인한다.
- 고정 C=1의 기존 표현도 사전 지정한 보조 비교로 보존한다.

## 주 평가

- 독립 검증 t=8에서 C − B NLL과 Brier 차이.
- 질문 단위 paired bootstrap 5,000회, 95% 구간.
- NLL이 개선되고 그 구간이 0 아래에 있으며 Brier가 점 추정상 악화하지 않을 때, 이번 조건에서 관계 정보의 추가 효과를 지지하는 결과로 판단한다.
- 구간이 0을 포함하면 불확실한 결과로 보고한다. 부정적 결과를 연구 전체의 불가능성으로 확장하지 않는다.
- 나머지 prefix와 모든 후보별 결과는 사전 지정한 보조 분석이다. holdout에서 가장 좋은 표현으로 주 결과를 교체하지 않는다.

## 보조 평가: 종료 정책 replay

- 비교할 prefix는 2/4/8/12/16으로 제한한다.
- nested OOF 개발 예측에서 threshold 0.9/0.95/0.975/0.99/0.995/조기 종료 없음 중 선택한다.
- 개발 다수결@16 대비 정확도 감소 1%p 이내에서 평균 생성 token이 가장 적은 정책을 선택한다. B와 C에 같은 규칙을 적용한다.
- 선택한 threshold도 lock에 포함한다.
- 독립 검증에서 정확도, 표본 수, 생성 token, 회복 가능한 조기 종료 오류를 보고한다.
- 이 replay는 생성 token 비용의 분석이다. 실제 온라인 serving 전체 시간 절감이나 PGM의 이론적 위험 보장을 입증하지 않는다. batched 특징 추출 overhead는 별도로 보고한다.

## 실행 순서와 산출물

1. 환경 분리·vLLM smoke·raw confidence 일치 검증.
2. 개발 응답 vLLM 생성, 다층 특징 추출, nested CV.
3. 선택 lock 저장 및 개발 보고서 작성.
4. 독립 검증 응답 vLLM 생성, 같은 특징 추출.
5. 고정한 모델의 holdout 평가와 종료 replay, 결과 보고서 작성.

설정은 `configs/stage2_vllm.json`, 모든 신규 산출물은 `artifacts/runs/gemma3_stage2_vllm_v1`에 둔다. 생성·특징 추출은 응답별 원자적 저장으로 재개할 수 있게 한다. 엔진 초기화·생성·특징 추출·분석 시간을 분리해 기록한다.

---

## Historical source: `STAGE3_PGM_EXPERIMENT_PLAN.md`

# PGM-ASC 구현 및 본 실험 계획

작성일: 2026-10-31 (KST). 초기 계획을 보존하며, 아래 실행 기록에 현재 진행 상태를 덧붙인다.

기준 문서: [수정 방법 및 초기 실험 계획](./PGM_ASC_revised_method_and_experiments.md).

## 1. 이번 단계의 결정

후보 partition mixture를 사용하는 조건부 PGM을 직접 구현한다. 초기 계획의 온라인 clustering, 학습형 반복 할인, OTHER, 확률 보정, 순차 종료를 유지한다. 생성은 현재의 vLLM과 정확한 생성 token ID에 대한 Transformers teacher forcing을 재사용한다.

직전 관계 특징 실험에서 추가 효과가 통계적으로 확인되지 않았다는 결과는 유지한다. 이번 단계는 H1이 성립했다고 가정하는 확장이 아니라, 실제 답 선택·종료를 수행하는 구조에서 H2와 H3을 직접 검증하는 실험이다. 효과가 없거나 불확실한 결과도 보고한다.

앞선 채팅 설명을 정정한다. 2단계 주 지표는 **t=8에서 현재 다수결 답이 맞는지에 대한 이진 예측**이었다. 응답별 정답 예측이나 t=16 후보 답 분포 평가가 아니었다. 이번 PGM은 후보 답 전체에 대한 분포를 학습한다.

## 2. 유지할 생성·특징 설정

| 항목 | 고정값 |
|---|---|
| 첫 모델 | `google/gemma-3-4b-it` |
| 모델 revision | `093f9f388b31de276ce2de164bdc2081324b9767` |
| MATH train revision | `21a5633873b6a120296cce3e2df9d5550074f4a3` |
| 생성 환경 | 기존 `.venv-vllm`, vLLM 0.18.0 |
| 특징·학습 환경 | 기존 `.venv`, Transformers 4.57.6 |
| GPU | 사용 가능한 B200 1장, TP=1; 실행 시 장치 점유 확인 |
| Precision | BF16 |
| Temperature / top-p / top-k | 0.7 / 0.95 / 해제 |
| 응답당 최대 새 토큰 | 4,096 |
| 최대 context | 8,192 |
| vLLM batching | 최대 128 sequences, 8,192 batched tokens, chunk 256 |
| GPU memory utilization | 0.60 |
| TF batching | 최대 8응답, padded token budget 16,384 |
| Prompt | 기존 공식 chat template와 `Solve the problem step by step. Put your final answer inside \boxed{}.` |
| 생성 시드 | 기존 base seed 42와 `(question_id, sample_index)` 해시 |
| 최대 응답 수 | 구현 점검 16, 본 실험 **32** |
| 본 실험 정책 확인 시점 | 모든 정수 prefix **1, 2, ..., 32** |
| 고정 예산 결과 | t=8, 16, 32 |
| 개별 confidence | temperature/top-p 변환 전 원모델의 log-probability·entropy |

LLM과 표현 encoder는 고정한다. 기존 답 추출·수식 동치성·먼저 등장한 답 우선 동률 규칙을 유지한다. 답 추출 실패와 truncation은 삭제하지 않는다. 새 생성에는 이전 답변을 prompt에 넣지 않는다.

## 3. PGM의 구체적인 정의

질문 q의 현재 관측을 다음과 같이 둔다.

$$
D_t=\{(r_i,a_i,x_i,e_i)\}_{i=1}^t,\qquad
\mathcal Y_t=C_t\cup\{\mathrm{OTHER}\}.
$$

여기서 x_i는 개별 confidence·길이·결측·truncation 특징, e_i는 추론 표현이다. 정답이 관측 답 집합 C_t에 없으면 학습 라벨은 OTHER이다. 정답 라벨은 clustering과 inference 함수에 전달하지 않는다.

### 3.1 온라인 partition: 최대 6개

처음에는 `final_full` 표현을 사용한다. 최종 boxed 답 이전의 reasoning token을 마지막 layer에서 평균 pooling한 벡터이며, 이미 응답별로 저장되어 있다.

추가 표현 후보는 학습 질문의 정규화 벡터에서 구한 고정 평균을 빼고 다시 L2 정규화한 표현이다. 평균은 질문마다 같은 가중치를 주어 train에서만 계산한다. 두 표현 중 하나를 dev NLL로 선택하며, 모든 관계 비교군에도 같은 선택 기회를 준다.

직전 실험의 `final_centered`는 매 prefix에서 평균이 달라지므로 과거 벡터도 변한다. 이것을 그대로 사용하면 기존 centroid와 고정 할당이 일치하지 않을 수 있다. 본 구현은 위 두 고정 변환만 사용해 과거 cluster 할당을 유지한다.

각 표현에 대해 다음 후보를 만든다.

| 후보 | 생성 규칙 |
|---|---|
| 1 | 모든 응답이 하나의 cluster |
| 2–5 | train 질문 내부 pair cosine의 10/40/70/90 분위수에서 정한 네 임계값의 온라인 centroid clustering |
| 6 | 모든 응답이 singleton cluster |

분위수 추정에는 train 질문마다 동일한 개수의 pair를 최대 128개까지 고정 시드로 표집하고, 전체 표본과 산출 임계값을 저장한다. dev/calibration/test의 유사도 분포로 임계값을 다시 계산하지 않는다.

임계값 기반 후보에서는 새 응답을 현재 centroid와 cosine이 가장 큰 cluster에 배정한다. 임계값 미만이면 새 cluster를 만들고, 동률이면 먼저 생성된 cluster를 선택한다. centroid는 해당 cluster의 고정 변환된 벡터 합을 정규화해 갱신한다. 과거 응답은 재할당하지 않는다. 표현이 결측인 응답은 이 후보들에서 singleton으로 처리하고 유사도 통계에서 제외한다. 강제 단일 cluster와 강제 singleton 후보는 각자의 정의를 유지한다.

각 prefix에서 동일한 할당 벡터는 한 번만 포함한다. gate는 중복 제거된 partition 집합 위에서 정규화한다. 따라서 같은 partition을 복제하는 것으로 prior mass가 늘어나지 않는다.

### 3.2 partition 확률과 답 지지 점수

partition m의 특징 v_m은 cluster 수, 크기 entropy, singleton 비율, cluster 내부 유사도와 centroid 간 유사도, 결측 mask로 제한한다. 답 문자열·답별 정오·gold는 입력하지 않는다.

$$
\pi_{t,m}=\frac{\exp f_\phi(v_m)}{\sum_j\exp f_\phi(v_j)},
\qquad
s_\theta(x_i)=\operatorname{softplus}(\mathrm{MLP}_\theta(x_i)).
$$

관측 답 y의 logit은 초기 계획대로 계산한다.

$$
S_m(y)=\sum_k n_{m,k}^{\alpha-1}
\sum_{i:z_i^{(m)}=k}s_\theta(x_i)\mathbf1[a_i=y],
\qquad \alpha=\sigma(\beta)\in(0,1).
$$

alpha는 전역 scalar 하나이며 초기값은 0.5이다. alpha=0과 alpha=1은 별도 고정 ablation으로 실행한다. 일반 후보에는 별도 빈도 bias를 넣지 않는다. cluster 크기에는 답 추출 실패도 포함하지만, 실패 응답은 관측 답에 지지 점수를 주지 않는다.

OTHER는 partition과 무관한 동일 head를 사용한다.

$$
S_m(\mathrm{OTHER})=u_\psi(w_t).
$$

w_t는 t, 답 종류 수·빈도·entropy, confidence 요약, 결측·truncation 비율이다. 관계 특징은 제외한다. 후보 분포를 출력하는 비-PGM 비교군에도 같은 입력과 head 구조를 사용하되 각 모델의 train loss로 함께 학습한다.

최종 분포는 logit 평균이 아니라 **확률분포의 mixture**다.

$$
p(y,z^{(m)}\mid D_t)=\pi_{t,m}\operatorname{softmax}_{b\in\mathcal Y_t}S_m(b)\big|_y,
$$

$$
p_t(y)=\sum_m\pi_{t,m}\operatorname{softmax}_{b\in\mathcal Y_t}S_m(b)\big|_y.
$$

계산은 log-softmax와 log-sum-exp로 구현한다. 이것은 제한된 후보 partition 위의 조건부 latent-variable model이며, decoder 응답 간 실제 생성 의존성이나 Bayesian likelihood를 복원했다고 해석하지 않는다.

### 3.3 답 선택과 학습

$$
\hat a_t=\arg\max_{a\in C_t}p_t(a),\qquad
\mathcal L=-\frac1{|Q_{\mathrm{train}}|}
\sum_q\frac1{32}\sum_{t=1}^{32}\log p_t(y^*_{q,t}).
$$

답 선택은 OTHER를 제외한 관측 답 중에서 한다. 답 동치 집합의 대표는 prefix 안에서 첫 등장한 답으로 정하고, 미래 응답을 사용해 과거 답 ID를 재정의하지 않는다. C_t가 비어 있으면 계속 생성하고, 최대 예산에서도 비어 있으면 추출 실패로 기록하며 정확도는 0으로 센다. OTHER가 전체 최빈 상태이면 조기 종료하지 않는다.

구현 점검은 t<=16에서만 수행한다. 본 실험에서는 1부터 32까지 모두 학습·보정하며, 16개로 학습한 정책을 검증 없이 32개로 늘리지 않는다.

| 학습 항목 | 초기 설정 |
|---|---|
| support / gate / OTHER head | 각각 hidden 32, GELU, 선형 출력 |
| 최적화 | AdamW, gradient clipping 1.0 |
| 학습률 후보 | 0.001, 0.003 |
| weight decay 후보 | 0.0001, 0.001 |
| 배치 | 질문 32개, 각 질문의 prefix loss를 먼저 평균 |
| Epoch | 최대 200, dev 평균 prefix NLL 기준 patience 20 |
| 연속 특징 표준화 | train 통계만 사용, 결측 mask 별도 |
| 모델 선택 | dev의 t=8/16/32 평균 NLL; 동률이면 작은 모델·빠른 checkpoint |
| 주 학습 시드 | 42 |
| 강건성 확인 | 선택된 구조·학습 설정을 유지하고 시드 43, 44로 재학습 |

주 비교는 seed 42로 고정한다. 추가 시드에서 더 좋은 모델로 주 결과를 바꾸지 않는다. 각 시드의 보정과 정책 선택은 같은 calibration 분할 안에서 별도로 수행한다. 질문 bootstrap과 학습 시드 간 변동을 분리해 보고한다.

## 4. 데이터 분할과 기존 bank의 재사용

첫 engineering run은 기존 vLLM 286문제 × 16응답 = **4,576응답**만 사용한다. 기존 dev 136문제와 이전 holdout 중 question ID 정렬 앞 64문제를 train 200문제로 사용하고, 나머지 86문제를 engineering dev로 둔다. 이미 본 자료이므로 성능 주장의 독립 평가로 사용하지 않는다.

본 실험은 다음 분할로 새 manifest를 고정한다.

| 역할 | 문항 수 | 용도 |
|---|---:|---|
| Train | 800 | 기존 286문제 + 새 514문제; 집계기·표현 변환·유사도 임계값 학습 |
| Development | 200 | 새 문제; 구조·표현·학습 설정·단일 partition 선택 |
| Calibration-A | 150 | 새 문제; 확률 temperature 보정 |
| Calibration-B | 150 | 새 문제; 전체 순차 정책의 종료 임계값 선택 |
| Final test | MATH-500 전체 500 | 초기 계획의 주 최종 benchmark, 설정 고정 후 평가 |

MATH train의 신규 1,014문제는 유효한 Level 1–5와 `[asy]` 없는 모집단에서 선택한다. split별 난이도 수를 균형화하고 그 안에서 과목 비중을 보존한다. train은 기존 286문제를 포함한 최종 난이도 균형을 맞추도록 추가 표집한다. 질문 ID와 공백 정규화한 문제 hash로 중복을 제거하고 이전 smoke 문항도 제외한다. 표집 seed는 20261031이다. 자격 문항 수를 확인한 뒤 부족하면 생성 전에 분할 규모 변경을 문서에 기록하며, 성능을 보고 조정하지 않는다.

기존 286문제는 모두 이미 결과를 봤으므로 본 실험에서 새 dev/calibration/test로 재지정하지 않는다. 최초 Transformers bank는 주 학습 bank에 섞지 않고 기존 분석 자료로 보존한다.

MATH-500은 전체 500개에 제공된 문제 텍스트를 그대로 사용한다. 도형 코드를 포함한 문항도 사후 삭제하지 않는다. `[asy]` 포함 여부를 생성 전에 표시하고 하위 집단 결과를 함께 보고한다. 이 조건은 도형 제외 train과 분포가 다르며, 이미지 이해를 평가하는 조건은 아니다. benchmark revision과 데이터 필드·중복 검사 결과는 test 생성 전 lock에 추가한다. 문제 텍스트만 사용하는 중복 검사는 허용하되 test 정답·solution은 평가 단계까지 분리한다.

## 5. 반드시 비교할 방법

첫 engineering run의 최소 비교와 본 실험 비교를 구분한다.

| 방법 | Engineering | 본 실험 | 검증할 효과 |
|---|:---:|:---:|---|
| SC@1/8/16/32 | 32 제외 | 포함 | 고정 예산 기준 |
| confidence 합산, 즉 alpha=1 | 포함 | 포함 | 관계 없이 개별 신뢰도만 사용 |
| cluster당 한 표 / alpha=0 패턴 평균 | 포함 | 포함 | 단순 중복 할인 |
| dev에서 고른 단일 partition + learned alpha | 포함 | 포함 | H2의 핵심 비교군 |
| uniform partition mixture | 포함 | 포함 | 학습 gate의 추가 가치 |
| learned partition mixture, 전체 PGM | 포함 | 포함 | 제안 방법 |
| 크기 보존 random partition mixture | 포함 | 포함 | grouping 의미와 단순 정규화 구분 |
| Count + confidence 이진 신뢰도 모델 | 포함 | 포함 | 직전 ASC 기준선의 재학습 버전 |
| 후보별 count + confidence MLP, OTHER 포함 | 이후 | 포함 | 동일 라벨로 학습한 후보 답 선택 기준선 |
| 동일 표현을 받는 DeepSets, OTHER 포함 | 이후 | 포함 | 표현·라벨과 PGM 구조의 효과 구분 |
| ASC / ESC / ReASC / CGES | 이후 | 포함 | 초기 계획의 기존 adaptive 비교군 |

본 실험 전 ASC/ESC/ReASC/CGES의 원 논문·공식 코드에 맞춰 구현 및 설정을 고정한다. 임의로 만든 빈도 threshold 정책을 해당 이름으로 보고하지 않는다. 비교군 재현이 준비되지 않으면 최종 benchmark 실행은 보류하고 engineering/dev 단계에서 계속 구현한다.

학습형 비교군은 같은 질문·라벨·prefix·calibration 분할을 사용한다. 후보 MLP는 답의 표면 문자열 대신 그 답을 지지한 경로의 count/confidence 요약을 입력한다. DeepSets도 후보 순서에 불변이 되게 구성하고, 답 ID는 동치 집합 소속으로만 제공한다. 파라미터 수와 탐색 trial 수를 함께 보고한다. 표현 두 종류 × 학습률 두 종류 × weight decay 두 종류의 최대 8개 설정을 관계 학습형 방법에 동일하게 허용한다.

single partition의 임계값 선택은 해당 방법의 구조 선택으로 dev에서만 수행하고 탐색 수를 별도로 기록한다. test에서 가장 좋은 partition을 고르는 oracle 비교로 대체하지 않는다. alpha=0/1, confidence 제거, gate 제거, 결론 구간 마스킹은 사전 지정 ablation으로 보고한다.

## 6. 평가를 세 실험으로 나누기

### A. 고정 예산 답 선택

같은 prefix에서 t=8/16/32의 accuracy, 후보 공간 multiclass NLL/Brier, 정답 미관측 비율과 OTHER 예측을 보고한다. Brier는 후보별 제곱 오차의 **합**으로 정의하며 후보 수로 나누지 않는다.

$$
\mathrm{NLL}_t=-\frac1Q\sum_q\log p_{q,t}(y^*_{q,t}),\qquad
\mathrm{Brier}_t=\frac1Q\sum_q\sum_{y\in\mathcal Y_{q,t}}
\left(p_{q,t}(y)-\mathbf1[y=y^*_{q,t}]\right)^2.
$$

H2의 주 비교는 learned mixture − dev-selected single partition의 **t=8/16/32 평균 NLL**이다. 두 방법에 같은 학습 분할·보정 절차를 적용하고, 보정 전후를 모두 보고한다. alpha=1 대비 비교와 accuracy는 사전 지정 보조 분석이다.

### B. 답 선택기를 고정한 종료 실험

모든 방법의 반환 답을 다수결 A_SC(D_t)로 고정한다. PGM은 그 다수결 답에 배정한 확률 p_t(A_SC(D_t))로 종료한다. 같은 calibration 절차로 보정한 count+confidence 신뢰도 모델과 비교한다. PGM이 최빈 후보에 주는 확률을 다수결 답의 확률처럼 사용하지 않는다.

이 실험에서는 후보 선택 개선이 섞이지 않은 조기 종료 차이를 확인한다. 다수결 답과 PGM이 선택한 답이 다른 prefix의 비율도 저장한다.

### C. 각 방법의 답 선택과 종료를 결합한 전체 시스템

calibration-A에서 T>0을 최적화한다.

$$
\tilde p_t(y)=\frac{p_t(y)^{1/T}}{\sum_b p_t(b)^{1/T}}.
$$

calibration-B에서 전체 prefix를 재생하며 gamma를 선택한다.

$$
\tau=\min\left(\{t:\tilde p_t(\hat a_t)\geq\gamma,
\;\hat a_t\in C_t,\;\mathrm{OTHER}\notin\arg\max_y\tilde p_t(y)\}\cup\{32\}\right).
$$

gamma 후보는 0.50/0.60/0.70/0.80/0.85/0.90/0.925/0.95/0.975/0.99/0.995와 '조기 종료 없음'이다. 종료 없음은 수치 임계값과 별도 flag로 저장한다. 각 방법에서 SC@32 대비 정확도 하락 1%p 이내인 정책 중 평균 생성 token이 최소인 것을 고른다. 정확도·비용이 동률이면 더 높은 gamma를 택한다. 유효한 adaptive 정책이 없으면 조기 종료 없음으로 고정하고 제약 미충족도 보고한다.

이진 confidence 정책은 이진 보정, 후보 분포 정책은 위 temperature 보정을 사용한다. 실험 B와 C의 보정 대상 및 선택된 threshold는 따로 저장한다. test에서 threshold를 재조정하지 않는다.

주 효율 결과는 이 고정 operating point의 반환 정확도, 평균 생성 token, 평균 응답 수와 실제 온라인 전체 시간이다. 가장 강한 adaptive 기준선은 test가 아니라 calibration-B의 같은 제약·비용 규칙으로 미리 지정한다. 정확도 차이와 신뢰구간을 비용 절감과 함께 제시한다. calibration의 1%p 허용 조건을 test 비열등성 보장으로 해석하지 않는다.

## 7. 통계와 비용 보고

질문 단위 paired bootstrap 5,000회로 방법 차이의 95% 구간을 계산한다. 같은 질문의 prefix, 답변, 순서 반복을 독립 표본으로 늘리지 않는다. H2 주 비교를 먼저 해석하고, 나머지 ablation은 보조 분석으로 명시한다.

반환 정확도는 실제 정답이 달라진 문제 수와 함께 보고한다. 종료 시점의 multiclass NLL/Brier, 반환 답의 binary calibration, 조기 종료율·조건부 오답률, 예산 소진율, 다음 회복 가능 오류를 기록한다.

$$
R_{\mathrm{recover}}=\Pr\{\tau<32,\ A(D_\tau)\ne y^*,\ A(D_{32})=y^*\}.
$$

A는 같은 방법의 동일 선택기다. 추가 sampling으로 정답이 오답으로 바뀌는 경우도 별도 집계한다.

replay의 생성 token 절감은 online 시간 절감과 구분한다. 생성, teacher forcing, 답 grouping, partition 갱신, head inference, CPU/GPU 전송, 메모리 peak를 분리 측정한다. 학습과 보정의 일회성 비용도 별도 보고한다. pair 검증에 사용한 gold 수식 동치성 비용은 정책의 추론 비용과 분리한다.

온라인 검증은 test 500개 중 **결과를 보기 전 고정 시드로 고른 100문제**에서 수행한다. SC32, 고정한 최강 adaptive 기준선, PGM을 같은 GPU와 question batch 8로 실행한다. 각 round에서 활성 질문마다 응답 하나를 생성하고 필요한 특징을 추출한 뒤 종료 질문을 제거한다. 첫 구현은 vLLM/TF를 같은 GPU 메모리 예산 안에서 함께 유지할 수 있는지 먼저 측정하고, 불가능하면 순차 초기화 비용까지 포함한다. 다른 방법에 불필요한 TF는 수행하지 않는다. warm-up 뒤 방법 실행 순서를 순환하며 3회 반복하고 평균과 범위를 보고한다. 별도 offline bank 생성 처리량으로 serving 이득을 주장하지 않는다.

## 8. 구현 순서와 통과 조건

| 순서 | 작업 | 완료 조건 / 산출물 |
|---|---|---|
| 1 | 기존 4,576응답용 prefix loader | 특징과 gold를 분리한 입력, question/response provenance |
| 2 | 온라인 clustering과 dedup | 모든 prefix의 cluster/centroid를 직접 검증 |
| 3 | support·gate·OTHER 및 mixture 학습 | 유한 loss, tiny synthetic bank overfit, checkpoint 재현 |
| 4 | 기존 bank의 engineering 비교 | 단일·uniform·learned mixture·alpha=0/1 결과와 collapse 분석 |
| 5 | 본 데이터 분할·기존 bank import | split hash, train-only 변환, import provenance lock |
| 6 | 최대 32응답 생성과 TF | 전체 train/dev/calibration bank 완성, 실패 포함 검증 |
| 7 | 비교군 구현·dev 선택 | 모든 설정과 모델 선택 기록 고정 |
| 8 | calibration-A/B 및 lock | checkpoint·표현·eta·T·gamma·benchmark revision·subset hash |
| 9 | MATH-500 평가 | 동일 bank에서 A/B/C 실험과 bootstrap, 설정 변경 없음 |
| 10 | 100문제 실제 온라인 검증 | 전체 시간·각 overhead·peak memory 비교 |

필수 구현 검증은 다음 성질을 직접 확인한다.

1. alpha=1이면 partition과 gate를 바꿔도 같은 분포가 나온다.
2. 모든 cluster가 singleton이면 alpha 값과 무관하게 같은 분포가 나온다.
3. 중복 partition 제거와 cluster ID 재명명은 출력에 영향을 주지 않는다.
4. mixture는 정상 확률분포이며 극단 logit에서도 유한하다.
5. 미래 응답과 gold를 변경해도 이미 계산한 prefix 출력은 변하지 않는다.
6. 정답이 미관측이면 OTHER로 학습하되 OTHER를 정답 텍스트로 반환하지 않는다.
7. 후보 답이 전혀 없는 경우, 답 동치 tie, zero vector, truncation이 정의된 대로 처리된다.
8. offline prefix 재생과 incremental online update의 출력이 일치한다.

gate entropy, 각 후보 weight, alpha, cluster 수, 후보 partition 중복률을 매 epoch와 평가 prefix별로 저장한다. alpha가 1로 수렴하거나 gate가 collapse하는 것을 실패로 숨기거나 강제로 방지하는 regularizer를 사후 추가하지 않는다. random partition은 cluster 크기를 보존하고 과거 응답을 재배정하지 않는 무작위 배정을 사용한다. 원 partition이 새 cluster를 만들면 같은 크기 1의 cluster를 추가하고, 기존 cluster에 추가할 때는 같은 현재 크기의 무작위 cluster를 선택한다. prefix별 cluster 크기 multiset을 맞추면서 유사도 기반 소속 관계를 제거한다.

## 9. 생성 규모와 예상 자원

첫 engineering run은 새 LLM 생성 없이 기존 bank로 수행한다. 본 Gemma 실험의 목표량은 다음과 같다.

| 항목 | 응답 수 |
|---|---:|
| Train/dev/calibration: 1,300 × 32 | 41,600 |
| 기존 vLLM bank에서 검증 후 재사용 | 4,576 |
| 이 분할에서 추가 생성 | 37,024 |
| MATH-500: 500 × 32 | 16,000 |
| 추가 offline 생성 합계 | **53,024** |

이미 저장된 16개 응답에 16개를 추가할 때 기존 sample index와 seed를 보존한다. 기존 split manifest는 수정하지 않는다. 새 manifest는 응답별 원래 fingerprint와 source path를 참조하고, 모델·prompt·decoding·token ID·추출 정의가 일치하는지 검증한다. 단순 복사 후 새로운 fingerprint인 것처럼 기록하지 않는다. benchmark token 길이에 따라 검증된 context 초과가 있으면 생성 전에 처리 방침을 기록하며, 조용히 prompt를 자르지 않는다.

직전 run의 평균 약 738 token/응답을 적용하면 추가 offline 출력은 약 3,913만 token이다. 당시 생성 처리량 약 7.5–7.8k token/s만 대입한 이상적인 생성 시간은 약 84–87분이다. TF, 모델 초기화, 학습, I/O, 비교군 재현, 온라인 반복 측정은 포함하지 않은 참고치다. N=32·데이터 구성·GPU 점유에 따른 실제 시간은 본 실행 초반 실측으로 갱신한다.

## 10. 후속 모델·과제와 주장 범위

Gemma에서 구현·평가 프로토콜을 고정한 뒤 초기 계획의 Qwen3.5-4B로 반복한다. 해당 모델의 revision, chat template, thinking mode, tokenizer·backend 호환성은 실행 전에 별도로 고정하고 thinking token도 비용에 넣는다. 이 단계에서 새 모델을 설치하거나 호환성을 검증했다고 간주하지 않는다.

GPQA Diamond는 초기 계획대로 수학에서 학습·보정한 정책의 외부 과제 전이 평가다. 선택지 전체를 후보에 넣고 OTHER를 사용하지 않는다. 비-PGM 비교군도 같은 전이 조건을 적용한다. 과학 정답을 이용한 재학습이나 보정을 수행하면 별도 in-domain 실험으로 보고한다. 다른 모델·과제로 전이할 때 정규화·gate·alpha·T·gamma를 모두 유지한 조건과 별도 calibration을 허용한 조건을 구분한다.

Gemma engineering run에서 먼저 확인할 것은 **mixture가 단일 partition 및 alpha=1보다 나아지는지**, 그리고 그 차이가 답 선택과 종료에서 나타나는지다. 평균 NLL 차이의 구간이 0을 포함하면 H2는 불확실하다. token 절감이 있어도 실제 online 전체 시간이 증가하면 H3의 시간 효율 개선은 확인되지 않은 것으로 보고한다. 주 결과는 MATH-500을 평가하기 전에 위 기준과 설정을 lock한 뒤 작성한다.

새 산출물은 `artifacts/runs/gemma3_pgm_v1/`과 `artifacts/data/math_pgm_v1/`에 저장할 예정이다. 예정 모듈은 `pgm_data.py`, `pgm_partitions.py`, `pgm_model.py`, `pgm_train.py`, `pgm_calibrate.py`, `pgm_replay.py`, `pgm_online.py`다. 이 문서의 모듈명은 구현 계획이며 현재 실행 가능한 CLI가 아니다.

## 11. 실행 기록

2026-10-31에 `pgm_data`, `pgm_partitions`, `pgm_model`, `pgm_train`, `pgm_experiment`를 구현했다. 실제 CLI는 README의 PGM 실행 절차를 따른다. 전체 프로젝트 테스트 25개를 통과했다. 첫 engineering 실험은 train 200문제·dev 86문제의 기존 bank에서 8개 학습형 방법으로 수행했다. 상수 특징의 표준화 스케일을 1로 고쳐 드문 답 추출 실패가 과도하게 확대되는 문제를 수정한 뒤 모든 모델을 다시 학습했다.

이 탐색 실험에서 mixture의 NLL@8은 0.18977, 할인 없는 alpha=1은 0.16840이었다. mixture와 고정한 단일 partition의 t=8/16 평균 NLL 차이는 +0.02308(95% CI [-0.02217, +0.07948])로, mixture 우위는 관측되지 않았다. engineering의 단일 partition은 후보 index 3으로 고정한 것이며, 본 실험에서 계획한 dev-selected single-partition 비교를 대체하지 않는다. 이 split에서는 보정이나 종료 정책을 선택하지 않았다.

본 분할은 계획대로 800/200/150/150문제로 고정했다. 기존 4,576응답의 생성 설정과 token/scoring provenance를 검증해 train에만 참조로 가져왔으며, 원래 기록의 fingerprint를 변경하지 않았다. GPU 4에서 N=32 본 bank 생성·TF driver를 시작했다. MATH-500은 로드·평가하지 않았다. 본 모델 선택, 보정, 기존 adaptive 비교군 재현과 온라인 시간 검증은 남아 있다.

- [Engineering 보고서](./artifacts/runs/gemma3_pgm_v1/engineering/report.md)
- [본 실험 현재 상태](./artifacts/runs/gemma3_pgm_v1/status.json)

2026-11-01 00:39 KST에 본 bank 41,600응답의 생성·특징 추출·검증이 모두 완료됐다. 답 추출 실패 69개와 truncation 62개를 삭제하지 않고 포함했다. 이후 GPU 4에서 train/dev의 104개 설정 비교와 A/B 보정을 수행하는 `pgm_main` 실행을 시작했다. 프로젝트 테스트 29개와 짧은 실제 CUDA 학습 검증을 통과했다. engineering의 DeepSets는 confidence만 받던 구현 점검 버전이었으며, 본 비교군에는 동일한 원래 reasoning hidden state도 입력하도록 구현했다. 원래 표현은 question별로 한 번 저장하고, 응답별 encoder를 적용한 뒤 현재 prefix mask로 pooling해 미래 응답의 영향이 없도록 검증했다.

- [본 모델 선택·보정 상태](./artifacts/runs/gemma3_pgm_v1/main/status.json)

2026-11-01 03:28 KST에 최종 protocol lock을 저장하고 MATH-500 수집을 시작했다. 그 전에 본 설정 104개, 추가 시드 16개, 비교군 5개의 보정을 완료했다. MATH-500 revision은 `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`다. 전체 500개를 보존하고, `[asy]`가 있는 42개를 생성 전에 표시했다. train/dev/calibration과 정규화 문제 중복은 0개이며 최대 prompt는 875 token이었다. test 정답은 생성·특징 추출 입력과 분리했다. GPU 4와 5에서 각 250문제 × 32응답을 처리한 뒤 자동으로 replay 평가를 수행한다. H2 주 비교는 보정 전 t=8/16/32 평균 multiclass NLL로 최종 lock에 명시했으며 보정 후 지표도 보고한다. 실제 온라인 시간 검증은 replay 이후 별도 단계다.

- [최종 평가 실행 상태](./artifacts/runs/gemma3_pgm_v1/final/status.json)

2026-11-01 03:59 KST에 MATH-500 수집 및 고정 정책 replay 평가를 완료했다. PGM 정확도 79.4%, CGES 80.2%, SC32 82.4%였다. PGM의 생성 token 비용은 12,318.5 token/문제, CGES는 3,594.5였다. H2 주 NLL 차이는 -0.01198(95% CI [-0.02903, +0.00524])로 유의한 개선이 확인되지 않았다. PGM은 SC32 대비 3%p 하락해 사전 허용한 1%p 조건을 test 점 추정에서 충족하지 못했다. 추가 시드 43/44의 test 보조 분석과 사전 표시한 도형 하위집단 분석도 완료했다.

사전에 선택한 100문제·배치 8·동일 GPU·3회 순환 순서의 실제 온라인 생성 시간 검증을 시작했다. 이 단계에서는 매 round 질문당 하나의 응답을 생성하는 공통 스케줄을 사용하며, 별도의 최적화된 SC32 bulk serving 성능을 주장하지 않는다. PGM에 필요한 실제 forward pass를 포함한다. GPU는 다른 작업과 공유하므로 반복 결과 및 조건을 함께 보고한다. 실제 시간 결과가 나오기 전에 시간 효율 개선을 주장하지 않는다.

- [MATH-500 결과 요약](./artifacts/runs/gemma3_pgm_v1/test/analysis/result_summary.md)
- [실제 온라인 시간 측정 상태](./artifacts/runs/gemma3_pgm_v1/online/status.json)

---

## Historical source: `artifacts/runs/gemma3_math_pilot_v1/analysis/report.md`

# Pilot 결과

- 독립 질문 수: 150
- 응답 수: 2400
- 답 추출 실패: 5
- 토큰 제한 종료: 1
- 진단용 실행 여부: False

| t | 예측기 | NLL | Brier | AUROC |
|---|---|---:|---:|---:|
| 2 | A_count | 0.3038 | 0.0866 | 0.7349 |
| 2 | B_count_confidence | 0.2828 | 0.0715 | 0.9190 |
| 2 | C_count_confidence_relation | 0.2801 | 0.0678 | 0.9190 |
| 2 | D_count_confidence_representation | 0.3916 | 0.0836 | 0.8910 |
| 4 | A_count | 0.2475 | 0.0699 | 0.8473 |
| 4 | B_count_confidence | 0.2232 | 0.0555 | 0.9389 |
| 4 | C_count_confidence_relation | 0.2402 | 0.0573 | 0.9322 |
| 4 | D_count_confidence_representation | 0.3482 | 0.0723 | 0.9073 |
| 8 | A_count | 0.3090 | 0.0622 | 0.8756 |
| 8 | B_count_confidence | 0.1706 | 0.0478 | 0.9443 |
| 8 | C_count_confidence_relation | 0.1891 | 0.0517 | 0.9365 |
| 8 | D_count_confidence_representation | 0.3082 | 0.0786 | 0.9125 |
| 12 | A_count | 0.3340 | 0.0771 | 0.8830 |
| 12 | B_count_confidence | 0.1905 | 0.0530 | 0.9343 |
| 12 | C_count_confidence_relation | 0.2004 | 0.0518 | 0.9259 |
| 12 | D_count_confidence_representation | 0.3544 | 0.0873 | 0.8880 |
| 16 | A_count | 0.3387 | 0.0772 | 0.8845 |
| 16 | B_count_confidence | 0.1978 | 0.0554 | 0.9343 |
| 16 | C_count_confidence_relation | 0.2053 | 0.0525 | 0.9238 |
| 16 | D_count_confidence_representation | 0.3523 | 0.0847 | 0.8931 |

## 주 비교: t=8, C − B

음수이면 관계 특징 추가가 유리하다. 신뢰구간은 질문 단위 bootstrap이다.

- nll: 0.01852, 95% CI [-0.01169, 0.05098]
- brier: 0.00386, 95% CI [-0.00640, 0.01416]

이 결과는 정답 확률 예측의 pilot이다. 실제 조기 종료의 정확도·비용 개선을 입증하는 실험과 구분한다.

---

## Historical source: `artifacts/runs/gemma3_math_pilot_v1/analysis/result_summary.md`

# Gemma-3-4B-IT 관계 정보 pilot 결과 보고

작성일: 2026-10-29. 분석 완료: 같은 날 19:48 한국시간.

## 핵심 결론

현재의 마지막 Transformer layer 평균 표현과 cosine 관계 특징에서는 답 빈도·confidence를 넘어서는 추가 효과가 확인되지 않았다. 주 비교인 t=8에서는 관계 특징을 추가한 모델의 NLL과 Brier score가 점 추정상 모두 악화했고, paired bootstrap 신뢰구간은 모두 0을 포함했다.

이는 현재 표현·특징·학습기·표본 범위의 결과다. PGM 전반이나 모든 reasoning 관계 표현의 효과가 없다는 결론은 아니다. PGM의 partition 주변화와 실제 adaptive stopping 비용 절감은 아직 평가하지 않았다.

## 실행과 데이터

- 모델: Gemma-3-4B-IT, 고정 revision, BF16.
- MATH train에서 난이도별 30개씩 150문제, 질문당 16응답: 총 2,400응답.
- 2/4/8/12/16응답 prefix에서 현재 다수결 답의 정답 여부를 예측했다.
- 질문 단위 5-fold: 각 fold 학습 120문제, 평가 30문제. 질문 중복 0.
- 표준화와 PCA는 학습 fold에서만 fit했다.
- 생성 엔진: Transformers/PyTorch 직접 decoding, B200 한 장, batch 16.
- 생성 token 수 1,827,547개, 생성 및 streaming 특징 수집 약 118.3분, 처리량 약 257.4 token/s.
- 답 추출 실패 5개(0.208%), 토큰 제한 종료 1개(0.042%). 해당 응답을 삭제하지 않았다.
- 마지막 batch 저장 후 특징 생성과 통계 분석은 약 10초가 걸렸다.

## 사전 지정 주 비교: t=8

아래는 최종 답 정확도가 아니라 **현재 다수결 답이 맞을 확률의 예측 품질**이다. 답 선택은 모든 예측기에서 동일하다. NLL과 Brier는 낮을수록, AUROC는 높을수록 좋다.

| 예측기 | 입력 | NLL | Brier | AUROC |
|---|---|---:|---:|---:|
| A | 빈도 | 0.3090 | 0.0622 | 0.8756 |
| B | 빈도 + confidence | **0.1706** | **0.0478** | **0.9443** |
| C | B + 관계 특징 | 0.1891 | 0.0517 | 0.9365 |
| D | B + 평균 표현의 PCA | 0.3082 | 0.0786 | 0.9125 |

C − B 차이에서 음수이면 관계 특징 추가가 유리하다.

| 지표 | 평균 차이 | 질문 단위 bootstrap 95% CI |
|---|---:|---|
| NLL | +0.01852 | [-0.01169, +0.05098] |
| Brier | +0.00386 | [-0.00640, +0.01416] |

두 신뢰구간 모두 0을 포함한다. 관계 특징의 개선이 입증되지 않았고, 유의한 악화라고 단정할 수도 없다. bootstrap은 고정한 OOF 모델의 질문별 손실을 재표집한 것으로 모든 학습 변동을 포함하는 구간은 아니다.

B는 A보다 점 추정상 개선됐지만, 별도 사후 계산한 B − A 구간도 NLL [-0.36679, +0.01089], Brier [-0.03869, +0.00817]로 0을 포함했다. confidence의 우월성 역시 이 작은 pilot만으로 통계적으로 확정하지 않는다.

## 다른 prefix

| 응답 수 t | B NLL | C NLL | C − B NLL | C − B Brier |
|---|---:|---:|---:|---:|
| 2 | 0.2828 | 0.2801 | -0.00269 | -0.00366 |
| 4 | 0.2232 | 0.2402 | +0.01702 | +0.00181 |
| 8 | 0.1706 | 0.1891 | +0.01852 | +0.00386 |
| 12 | 0.1905 | 0.2004 | +0.00986 | -0.00113 |
| 16 | 0.1978 | 0.2053 | +0.00751 | -0.00283 |

모든 prefix에서 C − B의 NLL/Brier 95% 구간이 0을 포함했다. t=2의 작은 점 추정 개선이나 t=12/16의 Brier 개선만 골라 긍정적 결론으로 사용하지 않는다.

## 실제 다수결 답 정확도

| 응답 수 | 정답 문제 수 | 정확도 |
|---|---:|---:|
| 2 | 128/150 | 85.33% |
| 4 | 128/150 | 85.33% |
| 8 | 129/150 | 86.00% |
| 12 | 132/150 | 88.00% |
| 16 | 132/150 | 88.00% |

이는 개발용으로 고른 MATH train 문항의 수치다. MATH-500 같은 최종 평가 세트의 benchmark 성능으로 해석하지 않는다.

## 추가 진단: 사후 분석

### 유사도의 분리력

- t=8에서 최빈 답 지지 경로 사이의 평균 cosine: 정답군 약 0.9847, 오답군 약 0.9861.
- 오답군 평균은 비교 가능한 경로 쌍이 있는 20문제, 정답군은 129문제에서 계산했다.
- 두 집단 모두 유사도가 매우 높은 구간에 모여 있다. 현재 pooled 표현이 문체·문제 맥락과 추론 구조를 잘 구별하는지는 추가 검토가 필요하다.
- 평균 차이만으로 통계적 분리력이나 인과 관계를 주장하지 않는다.

### 잘못된 만장일치

- t=8의 만장일치 질문 104개 중 오답 2개.
- 두 사례 모두 [asy] 도형 코드가 포함된 문제였다. 현재는 도형 이미지 없이 코드를 텍스트로 입력했다.
- 경로 개수 문제에서는 8개 응답이 모두 1330을 답했지만 정답은 160이었다. B의 정답 확률 예측은 약 99.907%, C는 약 99.978%로 관계 특징이 이 잘못된 합의를 완화하지 못했다.
- 음영 넓이 문제에서는 모든 응답이 -14.8을 답했고 정답은 8.6이었다. 답 추출·수치 정규화 오류가 아닌 실제 오답임을 원문과 대조했다.

### 도형 입력 여부

| 그룹 | 질문 수 | t=8 다수결 정답 | B NLL | C NLL |
|---|---:|---:|---:|---:|
| Asymptote 코드 포함 | 14 | 8/14 | 0.7905 | 0.9185 |
| Asymptote 코드 없음 | 136 | 121/136 | 0.1068 | 0.1141 |

이 표는 원래 OOF 예측을 사후에 나눈 기술 통계다. 그룹별 재학습을 하지 않았고 주 분석 표본도 변경하지 않았다. 코드 없는 부분에서도 C의 NLL 점 추정 개선은 관측되지 않았다.

## 다음 실험에 대한 판단

1. 지금 결과로 복잡한 PGM 확대를 정당화하지 않는다. 현재 응답 bank로 표현과 관계 특징을 먼저 점검한다.
2. 실패 사례를 검토하고 최종 layer 평균이 공유 오류보다 질문·문체를 반영하는지 확인한다. 중간 layer, 단계별 표현, 결론 마스킹을 후보로 비교하되 같은 데이터에서의 추가 분석은 탐색적 결과로 분류한다.
3. 관계 특징의 추가 차원 때문에 생기는 과적합 가능성을 개발 질문 안의 nested CV와 정규화 강도 비교로 확인한다. 현재 C=1 고정 결과는 그대로 보존한다.
4. 다음 독립 검증 표본의 텍스트·도형 입력 프로토콜을 미리 정한다. 두 만장일치 오답을 결과를 본 뒤 임의 제외하지 않는다.
5. 추가 생성이 필요하면 vLLM 생성과 batched teacher-forced 특징 추출을 분리한 경로를 먼저 검증한다. 현재 응답 bank는 보존한다.

## 근거 파일

- [원래 자동 생성 보고서](./report.md)
- [전체 지표와 주 비교 신뢰구간](./evaluation.json)
- [Out-of-fold 예측](./oof_predictions.jsonl)
- [질문 단위 fold 목록](./cv_folds.json)
- [도형 입력 여부별 사후 통계](./diagram_descriptive.json)
- [B와 A의 추가 비교](./additional_baseline_comparison.json)
- [생성 및 분석 실행 로그](../pilot.log)

---

## Historical source: `artifacts/runs/gemma3_stage2_vllm_v1/report.md`

# vLLM 2단계 실험 결과

- 독립 검증 문항: 150개
- 사전 지정 H1 판단: 추가 효과 확인되지 않음
- 선택 lock: `de95e3f4e2aa7adec95c4ed852beba5f56988c34e2a5b8a9ca8f32cb039453a4`

## 구현과 입력

생성은 vLLM 0.18.0, 특징 추출은 정확한 생성 token ID를 사용하는 batched causal teacher forcing이다. 원래 Transformers pilot bank는 보존했다. 개발과 독립 검증 모두 동일 vLLM backend에서 생성했다.

개발은 기존 pilot 중 Asymptote 코드가 없는 136문항, 독립 검증은 겹치지 않는 새 텍스트 문항 150개다. 모두 MATH train에 속하며 최종 benchmark 평가와 구분한다.

## 독립 검증 주 비교: t=8

현재 다수결 답의 정답 확률 예측 지표다. 답 선택 자체는 모든 예측기에서 같다.

| 방법 | 선택 표현 | C | NLL | Brier | AUROC |
|---|---|---:|---:|---:|---:|
| baseline | baseline | 0.1 | 0.16513 | 0.04772 | 0.9209 |
| selected_relation | final_centered | 0.1 | 0.15535 | 0.04347 | 0.9215 |
| original_relation_tuned | final_full | 0.1 | 0.15471 | 0.04293 | 0.9215 |
| baseline_fixed_C1 | baseline | 1.0 | 0.18813 | 0.05080 | 0.9227 |
| original_relation_fixed_C1 | final_full | 1.0 | 0.16859 | 0.04241 | 0.9191 |

selected_relation − baseline: 음수가 관계 특징에 유리하다.

- nll: -0.00978, 질문 bootstrap 95% CI [-0.02460, +0.00302]
- brier: -0.00424, 질문 bootstrap 95% CI [-0.00960, +0.00039]

## 개발 nested CV

아래는 outer fold 밖에서 예측한 개발 결과다. 각 outer 학습 집합 안에서만 inner CV로 표현과 C를 선택했다.

| 방법 | NLL | Brier | AUROC |
|---|---:|---:|---:|
| baseline | 0.18881 | 0.06116 | 0.9485 |
| selected_relation | 0.19126 | 0.06555 | 0.9491 |
| original_relation_tuned | 0.18736 | 0.06274 | 0.9491 |

## 고정 종료 정책의 독립 검증 replay

threshold는 개발 nested OOF에서 선택해 lock에 저장했다. 표는 생성 token 비용의 replay이며 실제 온라인 serving 전체 시간 절감을 의미하지 않는다.

| 방법 | threshold | 정확도 | 평균 응답 수 | 생성 token 절감 | 회복 가능한 조기 오류 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.9 | 92.00% | 4.23 | 57.12% | 0 |
| selected_relation | 0.9 | 92.00% | 4.25 | 56.39% | 0 |

독립 검증 다수결@16 정확도는 92.00%다. 개발에서의 1%p 허용 조건이 독립 검증에서도 자동 보장되지는 않는다.

## 실측 시간

| 분할 | vLLM 초기화(초) | 생성(초) | 생성 token/s | 특징 추출(초) |
|---|---:|---:|---:|---:|
| smoke | 82.1 | 11.8 | 699.4 | 1.4 |
| dev | 36.3 | 213.2 | 7546.8 | 79.3 |
| holdout | 38.2 | 227.6 | 7773.2 | 84.8 |

## 해석 범위

- 원래 pilot과 생성 backend·모집단이 다르므로 기존 수치와 새 수치의 차이를 표현 개선만의 인과 효과로 해석하지 않는다. 새 실험 안의 B/C paired 비교가 주 결과다.
- early80은 고정 구간 마스킹이며 논리적 오류를 직접 판별한 결과가 아니다.
- bootstrap 구간은 고정한 학습 모델에 대한 질문 표집 변동을 나타내며 모든 학습 변동을 포함하지 않는다.
- PGM partition 주변화의 효과와 이론적 종료 위험 보장은 아직 이 실험의 평가 대상이 아니다.
- 독립 검증 결과를 보고 표현·C·threshold를 다시 선택하지 않았다.

## 파일

- [사전 계획](../../../STAGE2_EXPERIMENT_PLAN.md)
- [선택 lock](./selection_lock.json)
- [독립 검증 전체 지표](./holdout/analysis/evaluation.json)
- [개발 nested fold와 선택 기록](./dev/analysis/nested_folds.json)

## 실행 검증

- 프로젝트 테스트 19개 통과.
- 개발·검증 문항 중복 0개, 모두 사전 정의한 Asymptote 제외 조건 충족.
- 모델 선택 lock이 holdout 생성 시작보다 먼저 저장됨을 확인.
- dev: 2176응답, 답 추출 실패 0개, 토큰 제한 종료 0개. 수치 특징은 모두 유한값.
- dev vLLM/teacher-forcing token log-probability: 응답별 평균 절대차의 전체 평균 0.00433, 최대 0.01399; 허용치 0.05 초과 0개.
- holdout: 2400응답, 답 추출 실패 3개, 토큰 제한 종료 1개. 수치 특징은 모두 유한값.
- holdout vLLM/teacher-forcing token log-probability: 응답별 평균 절대차의 전체 평균 0.00438, 최대 0.01118; 허용치 0.05 초과 0개.
- 자동 파이프라인 경과 시간 약 13.41분. 환경 설치와 별도 최초 smoke 실행 시간은 포함하지 않음.
- [전체 실행 검증 기록](./execution_audit.json)

---

## Historical source: `artifacts/runs/gemma3_pgm_v1/completion_summary.md`

# PGM v1 completion

MATH-500 replay and all nine prespecified online timing runs are complete. Original primary models, locks and implementation hashes remain unchanged.

- [Frozen replay results](./test/analysis/result_summary.md)
- [Actual online timing](./online/report.md)
- [v2 follow-up](../gemma3_pgm_v2/result_summary.md)

The v1 online mean total times were SC32 3,481.2 s, CGES 897.4 s, and PGM 3,118.5 s, measured on the same 100 questions with three rotating-order repetitions and group-of-eight scheduling. These timings use the original v1 policy and do not verify v2 latency.

---

## Historical source: `artifacts/runs/gemma3_pgm_v1/engineering/report.md`

# PGM engineering experiment

Existing, previously inspected bank; exploratory results only.

Train 200 questions, engineering dev 86; 16 responses per question.

| Method | Acc@8 | NLL@8 | Acc@16 | NLL@16 | alpha |
|---|---:|---:|---:|---:|---:|
| alpha1 | 0.9302 | 0.16840 | 0.9419 | 0.26525 | 1.0000 |
| alpha0 | 0.9302 | 0.18871 | 0.9302 | 0.30443 | 0.0000 |
| single | 0.9419 | 0.19201 | 0.9302 | 0.25521 | 0.5015 |
| uniform | 0.9302 | 0.18256 | 0.9302 | 0.28421 | 0.5035 |
| mixture | 0.9302 | 0.18977 | 0.9302 | 0.30362 | 0.5030 |
| random | 0.9302 | 0.18944 | 0.9302 | 0.30560 | 0.5020 |
| candidate_mlp | 0.9302 | 0.17751 | 0.9302 | 0.30426 | 0.5000 |
| deepsets | 0.9302 | 0.19551 | 0.9302 | 0.35797 | 0.5000 |

SC accuracy: {'1': 0.872093023255814, '8': 0.9302325581395349, '16': 0.9186046511627907}

Mixture minus fixed engineering single-partition mean NLL: {'difference': 0.023083698004484177, 'ci95': [-0.022170915361493824, 0.0794833555817604], 'questions': 86, 'repetitions': 5000}

No independent H2/H3 claim; no probability calibration or stopping selection on this engineering split.

---

## Historical source: `artifacts/runs/gemma3_pgm_v1/main/baselines/report.md`

# Adaptive baselines: calibration results

Calibration-B used for policy selection; not test results.

| Method | Accuracy | Mean samples | Mean tokens | gamma |
|---|---:|---:|---:|---:|
| asc | 0.8667 | 3.140 | 3646.3 | 0.8 |
| esc_w4 | 0.8667 | 7.360 | 9019.0 | 1.0 |
| reasc | 0.8667 | 2.720 | 3483.8 | 0.8 |
| cges | 0.8733 | 2.787 | 3052.5 | 0.975 |
| count_confidence | 0.8667 | 6.387 | 9657.1 | 0.925 |

ESC uses disjoint blocks of four and the unanimous block answer on early stop (original Algorithm 1).
ReASC uses full-vocabulary bottom-10%/128 confidence, lambda=.7, gate target=.9, calibration-A first responses.
CGES selects geometric vs arithmetic LNS using DEV only and recomputes earlier evidence at current candidate count.
The 150-question calibration size and shared calibration-B threshold grid follow this study's protocol.

---

## Historical source: `artifacts/runs/gemma3_pgm_v1/main/report.md`

# PGM main development results

Development model selection only; no final benchmark evaluated.

Train 800, development 200, calibration-A 150, calibration-B 150; N=32.

| Method | Representation | Acc@8 | Acc@16 | Acc@32 | Mean NLL@8/16/32 | alpha |
|---|---|---:|---:|---:|---:|---:|
| mixture | final_full | 0.9000 | 0.8950 | 0.9050 | 0.28321 | 0.4924 |
| uniform | final_full | 0.9000 | 0.8950 | 0.9050 | 0.28648 | 0.4927 |
| alpha1 | final_full | 0.9000 | 0.9000 | 0.9100 | 0.31104 | 1.0000 |
| alpha0 | final_full | 0.9000 | 0.8950 | 0.9050 | 0.27575 | 0.0000 |
| random | final_full | 0.9000 | 0.8950 | 0.9050 | 0.28251 | 0.4911 |
| candidate_mlp | final_full | 0.9050 | 0.8950 | 0.9000 | 0.26459 | 0.5000 |
| deepsets | final_full | 0.9000 | 0.9000 | 0.9000 | 0.33898 | 0.5000 |
| single | final_full | 0.9050 | 0.8950 | 0.9050 | 0.29067 | 0.5270 |

These development scores were used for selection and cannot establish independent H2/H3 gains.

Probability temperatures and stopping thresholds were fitted on separate calibration sets.
Final benchmark evaluation is gated on existing adaptive baseline reproduction and the final protocol lock.

---

## Historical source: `artifacts/runs/gemma3_pgm_v1/online/report.md`

# Actual online time validation

Prespecified 100 questions, batch eight, same GPU, three rotating-order repetitions.

| Method | Mean total seconds | Min–max seconds | Mean accuracy | Mean samples |
|---|---:|---:|---:|---:|
| sc32 | 3481.21 | 3464.48–3508.44 | 0.8333 | 32.000 |
| cges | 897.36 | 855.45–922.28 | 0.8033 | 3.407 |
| mixture | 3118.51 | 3111.44–3126.80 | 0.8300 | 8.967 |

Includes live vLLM calls and PGM's teacher forcing. Model load/warmup are one-time costs reported separately.
GPU is shared with other processes; the three repetitions describe this execution environment, not an isolated hardware benchmark.

---

## Historical source: `artifacts/runs/gemma3_pgm_v1/test/analysis/improvement_diagnostics/report.md`

# PGM improvement diagnostics

Post-hoc analysis of the existing bank; no new primary result or independent validation.

## Stopping and cost

{'questions': 500, 'wrong': 103, 't1_stops': 318, 't1_wrong': 22, 'budget_exhausted': 98, 'budget_exhausted_wrong': 60, 'budget_exhausted_token_fraction': 0.8704016783913446, 'wrong_gold_unseen_by_32': 36, 'wrong_gold_unseen_at_stop': 62, 'wrong_gold_seen_at_stop': 41, 'recoverable_early_errors': 19, 'wrong_mean_partition_information': 0.12054070085287094, 'correct_mean_partition_information': 0.023987622931599617, 'late_improvement_among_budget_exhausted': {'8': {'budget_exhausted_correct_at_t': 35, 'budget_exhausted_correct_at_32': 38, 'budget_exhausted_corrected_by_32': 5, 'budget_exhausted_regressed_by_32': 2}, '16': {'budget_exhausted_correct_at_t': 39, 'budget_exhausted_correct_at_32': 38, 'budget_exhausted_corrected_by_32': 3, 'budget_exhausted_regressed_by_32': 4}}}

## Gate and partition effects

| t | Singleton mass | All-one mass | Mean unique partitions | Partition JS | Partition answer disagreement |
|---|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 1.0000 | 1.00 | 0.000000 | 0.0000 |
| 2 | 0.7947 | 0.2053 | 2.00 | 0.000817 | 0.0000 |
| 4 | 0.6938 | 0.1292 | 3.17 | 0.004324 | 0.0320 |
| 8 | 0.5534 | 0.1327 | 4.07 | 0.015263 | 0.0480 |
| 16 | 0.4047 | 0.1836 | 4.57 | 0.038299 | 0.0600 |
| 32 | 0.2140 | 0.3271 | 4.90 | 0.064235 | 0.0400 |

## OTHER calibration shift

| t | Cal-B actual unseen | Cal-B predicted OTHER | Test actual unseen | Test predicted OTHER |
|---|---:|---:|---:|---:|
| 1 | 0.1667 | 0.1672 | 0.2680 | 0.2104 |
| 2 | 0.1133 | 0.1265 | 0.1900 | 0.1563 |
| 4 | 0.0933 | 0.1024 | 0.1500 | 0.1315 |
| 8 | 0.0933 | 0.0767 | 0.1160 | 0.0989 |
| 16 | 0.0733 | 0.0657 | 0.0940 | 0.0853 |
| 32 | 0.0600 | 0.0682 | 0.0720 | 0.0934 |

## Controlled policy ablations

Fixed original model, T and gamma. Minimum, patience and hard cap changed only in replay.

| Minimum | Patience | Cap | Accuracy | Mean samples | Mean tokens | Recoverable errors |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 32 | 0.8160 | 9.068 | 13198.8 | 8 |
| 2 | 2 | 32 | 0.8240 | 9.736 | 14164.9 | 4 |
| 2 | 3 | 32 | 0.8240 | 10.904 | 15186.1 | 3 |
| 2 | 1 | 16 | 0.8180 | 5.700 | 7629.1 | 12 |
| 2 | 1 | 8 | 0.8100 | 3.804 | 4556.1 | 15 |
| 4 | 1 | 32 | 0.8220 | 10.796 | 14635.7 | 4 |
| 4 | 2 | 32 | 0.8240 | 11.258 | 15209.6 | 3 |
| 4 | 3 | 32 | 0.8260 | 11.600 | 15622.0 | 2 |
| 4 | 1 | 16 | 0.8240 | 7.264 | 8765.4 | 8 |
| 4 | 1 | 8 | 0.8160 | 5.246 | 5508.3 | 11 |

Random control: {'8': {'answer_agreement': 0.988, 'mean_total_variation': 0.0032409525010734797}, '16': {'answer_agreement': 0.994, 'mean_total_variation': 0.004464628174901009}, '32': {'answer_agreement': 0.994, 'mean_total_variation': 0.004314694553613663}}

Pair cosine (descriptive, not independent-pair inference): {'all': {'pairs': 245625, 'mean': 0.9831052422523499, 'std': 0.018266135826706886, 'quantiles': [0.9639079809188843, 0.9886245727539062, 0.9966615676879883]}, 'same_answer': {'pairs': 191847, 'mean': 0.9849197864532471, 'std': 0.017256632447242737, 'quantiles': [0.9683400988578796, 0.9899953603744507, 0.9971176862716675]}, 'different_answer': {'pairs': 53778, 'mean': 0.9766320586204529, 'std': 0.02019733190536499, 'quantiles': [0.9523000597953797, 0.9826965034008026, 0.9933428347110749]}}

The zero-event binomial upper bound refers to the harmful-disagreement event, not a direct signed-accuracy noninferiority proof.
Frozen-head alpha sensitivity is not a retrained alpha ablation; test values must not be used to select a new primary alpha.

---

## Historical source: `artifacts/runs/gemma3_pgm_v1/test/analysis/min_samples_ablation/report.md`

# Minimum-sample post-hoc ablation

The test benchmark has already been inspected. This is exploratory replay, not a new independent validation.

## PGM with original T and gamma=0.85

| min_samples | Accuracy | Mean samples | Mean tokens | Recoverable early errors | Corrected / regressed |
|---|---:|---:|---:|---:|---:|
| 1 | 79.4% | 7.944 | 12318.5 | 19 | 0 / 0 |
| 2 | 81.6% | 9.068 | 13198.8 | 8 | 11 / 0 |
| 4 | 82.2% | 10.796 | 14635.7 | 4 | 15 / 1 |
| 8 | 82.8% | 13.792 | 16753.1 | 1 | 18 / 1 |

All methods, including CGES, are subjected to the same minimum grid in evaluation.json.
A second arm reselects gamma using calibration-B only for each minimum; temperature and model remain fixed.
Increasing the minimum guarantees more observations, not correctness. The original online policy is unchanged.

Illustrative 4^x=8 question: {'1': {'question_id': 'math500_0048_c0cb53c3', 't': 1, 'chosen': 0, 'correct': False, 'token_cost': 604, 'recoverable_error': True}, '2': {'question_id': 'math500_0048_c0cb53c3', 't': 6, 'chosen': 1, 'correct': True, 'token_cost': 3250, 'recoverable_error': False}, '4': {'question_id': 'math500_0048_c0cb53c3', 't': 6, 'chosen': 1, 'correct': True, 'token_cost': 3250, 'recoverable_error': False}, '8': {'question_id': 'math500_0048_c0cb53c3', 't': 8, 'chosen': 1, 'correct': True, 'token_cost': 4612, 'recoverable_error': False}}

---

## Historical source: `artifacts/runs/gemma3_pgm_v1/test/analysis/report.md`

# Locked MATH-500 results

500 questions, 32 responses per question; test settings fixed before generation.

| Method | Accuracy | Mean samples | Mean generated tokens |
|---|---:|---:|---:|
| mixture | 0.7940 | 7.944 | 12318.5 |
| uniform | 0.7940 | 8.040 | 12436.3 |
| alpha1 | 0.8000 | 8.076 | 12565.7 |
| alpha0 | 0.7920 | 7.230 | 11423.5 |
| random | 0.7960 | 8.076 | 12484.6 |
| candidate_mlp | 0.8220 | 17.456 | 21764.8 |
| deepsets | 0.8000 | 8.722 | 13394.6 |
| single | 0.8060 | 9.364 | 14199.7 |
| asc | 0.8020 | 3.718 | 4831.8 |
| esc_w4 | 0.8220 | 10.608 | 14091.4 |
| reasc | 0.7980 | 3.392 | 4694.3 |
| cges | 0.8020 | 3.014 | 3594.5 |
| count_confidence | 0.8000 | 9.566 | 14550.3 |
| sc1 | 0.7320 | 1.000 | 899.0 |
| sc8 | 0.8020 | 8.000 | 7172.4 |
| sc16 | 0.8220 | 16.000 | 14315.5 |
| sc32 | 0.8240 | 32.000 | 28602.0 |

Primary H2, mixture minus single mean NLL: {'difference': -0.011976263228136531, 'ci95': [-0.029032884869286343, 0.005239405774788286], 'questions': 500, 'repetitions': 5000}

These token savings are replay results. Online serving time has not yet been verified.

Preselected strongest adaptive baseline: cges

Paired accuracy/cost contrasts: {'correct': {'difference': -0.008, 'ci95': [-0.028049999999999985, 0.012], 'questions': 500, 'repetitions': 5000}, 'token_cost': {'difference': 8723.96, 'ci95': [7163.9387, 10389.079549999999], 'questions': 500, 'repetitions': 5000}}

---

## Historical source: `artifacts/runs/gemma3_pgm_v1/test/analysis/result_summary.md`

# MATH-500 result summary

The locked primary replay evaluation and prespecified supplementary analyses are complete.

| Method | Accuracy | Mean answers | Mean generated tokens |
|---|---:|---:|---:|
| mixture | 79.4% | 7.944 | 12318.5 |
| single | 80.6% | 9.364 | 14199.7 |
| candidate_mlp | 82.2% | 17.456 | 21764.8 |
| asc | 80.2% | 3.718 | 4831.8 |
| esc_w4 | 82.2% | 10.608 | 14091.4 |
| reasc | 79.8% | 3.392 | 4694.3 |
| cges | 80.2% | 3.014 | 3594.5 |
| count_confidence | 80.0% | 9.566 | 14550.3 |
| sc32 | 82.4% | 32.000 | 28602.0 |

PGM's accuracy is 3.0 percentage points below SC32, exceeding the prespecified 1-point point-estimate tolerance.
The calibration constraint did not transfer to this test distribution. CGES also falls below that tolerance; this is not a claim of matched-accuracy superiority.

Primary mixture minus single-partition NLL: {'difference': -0.011976263228136531, 'ci95': [-0.029032884869286343, 0.005239405774788286], 'questions': 500, 'repetitions': 5000}
The interval includes zero, so H2 is not established.

PGM minus preselected CGES: {'correct': {'difference': -0.008, 'ci95': [-0.028049999999999985, 0.012], 'questions': 500, 'repetitions': 5000}, 'token_cost': {'difference': 8723.96, 'ci95': [7163.9387, 10389.079549999999], 'questions': 500, 'repetitions': 5000}}
PGM has higher generated-token cost and does not establish an accuracy benefit. These costs are replay, not measured serving time.

## Extra training seeds

Settings and the primary seed were fixed before test evaluation.

| Method | Seed42 mean NLL | Seed43 mean NLL | Seed44 mean NLL |
|---|---:|---:|---:|
| mixture | 0.49150 | 0.49174 | 0.49158 |
| single | 0.50347 | 0.50966 | 0.50927 |
| random | 0.49128 | 0.49150 | 0.49352 |
| candidate_mlp | 0.50818 | 0.50808 | 0.49499 |

## Diagram subgroup

All 500 questions are retained. Diagram code is provided as text; no rendered images were supplied.

| Subgroup | Questions | PGM accuracy | CGES accuracy | SC32 accuracy |
|---|---:|---:|---:|---:|
| asy | 42 | 50.0% | 59.5% | 57.1% |
| without_asy | 458 | 82.1% | 82.1% | 84.7% |

## Actual online timings

Prespecified 100 questions, batch eight, three rotating-order repetitions are a separate live-generation experiment. See `online/status.json`; no online time gain is claimed here.

![Fixed-policy accuracy and generated-token cost](./supplementary/accuracy_token_tradeoff.png)

---

## Historical source: `artifacts/runs/gemma3_pgm_v3/report.md`

# PGM v3 mechanism experiment

500 fresh questions. All model and policy choices locked before holdout evaluation.

![Accuracy and generated tokens](./accuracy_tokens.png)

Primary PGM: `likelihood_math_quality_gate`. Matched random: `likelihood_math_random_quality_gate`.

Primary raw mean NLL contrast at t=2/4/8/16/32:

{'difference': 0.0007200550181223662, 'ci95': [-0.0006882459814609319, 0.002164664734622963], 'questions': 500, 'repetitions': 5000}

| Model | Accuracy | Mean samples | Mean generated tokens | SC32-minus-1pp point constraint |
|---|---:|---:|---:|---|
| likelihood_math_quality_gate | 0.7860 | 11.290 | 16446.5 | True |
| likelihood_math_random_quality_gate | 0.7860 | 11.300 | 16447.9 | True |
| likelihood_math_independent | 0.7880 | 7.860 | 11591.7 | True |
| likelihood_math_single0 | 0.7820 | 11.782 | 17210.9 | True |
| cges | 0.7740 | 6.430 | 8019.5 | True |
| asc | 0.7760 | 3.588 | 4542.6 | True |
| reasc | 0.7780 | 3.450 | 4176.0 | True |
| frozen_v2 | 0.7760 | 2.888 | 3365.5 | True |
| sc8 | 0.7780 | 8.000 | 7328.8 | True |
| sc16 | 0.7720 | 16.000 | 14631.2 | True |
| sc32 | 0.7760 | 32.000 | 29301.5 | True |

## Primary comparisons

{'likelihood_math_random_quality_gate': {'correct': {'difference': 0.0, 'ci95': [0.0, 0.0], 'questions': 500, 'repetitions': 5000}, 'token_cost': {'difference': -1.352, 'ci95': [-159.87529999999998, 172.43445], 'questions': 500, 'repetitions': 5000}}, 'likelihood_math_single0': {'correct': {'difference': 0.004, 'ci95': [-0.006, 0.014], 'questions': 500, 'repetitions': 5000}, 'token_cost': {'difference': -764.384, 'ci95': [-1291.5270999999998, -297.44425000000035], 'questions': 500, 'repetitions': 5000}}, 'likelihood_math_independent': {'correct': {'difference': -0.002, 'ci95': [-0.014, 0.008], 'questions': 500, 'repetitions': 5000}, 'token_cost': {'difference': 4854.874, 'ci95': [3774.33435, 6041.791049999999], 'questions': 500, 'repetitions': 5000}}, 'cges': {'correct': {'difference': 0.012, 'ci95': [-0.004, 0.028], 'questions': 500, 'repetitions': 5000}, 'token_cost': {'difference': 8427.076, 'ci95': [7082.74755, 9813.231399999999], 'questions': 500, 'repetitions': 5000}}, 'sc32': {'correct': {'difference': 0.01, 'ci95': [-0.006, 0.026], 'questions': 500, 'repetitions': 5000}, 'token_cost': {'difference': -12854.922, 'ci95': [-13939.33005, -11741.9156], 'questions': 500, 'repetitions': 5000}}, 'reasc': {'correct': {'difference': 0.008, 'ci95': [-0.01, 0.026], 'questions': 500, 'repetitions': 5000}, 'token_cost': {'difference': 12270.574, 'ci95': [10566.762550000001, 14013.452099999999], 'questions': 500, 'repetitions': 5000}}}

## All seed-42 arms

| Arm | Raw selection NLL | Acc@8 | Acc@32 |
|---|---:|---:|---:|
| positive_raw | 0.52860 | 0.7840 | 0.7900 |
| positive_equation | 0.52388 | 0.7820 | 0.7880 |
| likelihood_base_independent | 0.73487 | 0.7820 | 0.7860 |
| likelihood_math_independent | 0.94632 | 0.7760 | 0.7860 |
| likelihood_math_raw | 0.57566 | 0.7820 | 0.7840 |
| likelihood_math_equation | 0.57928 | 0.7800 | 0.7840 |
| likelihood_math_random | 0.57876 | 0.7800 | 0.7840 |
| likelihood_math_single0 | 0.58837 | 0.7860 | 0.7820 |
| likelihood_math_single1 | 0.63838 | 0.7900 | 0.7780 |
| likelihood_math_single2 | 0.72878 | 0.7720 | 0.7880 |
| likelihood_math_single3 | 0.85696 | 0.7700 | 0.7780 |
| likelihood_math_single4 | 0.92900 | 0.7760 | 0.7860 |
| likelihood_math_single5 | 0.94632 | 0.7760 | 0.7860 |
| likelihood_math_quality_gate | 0.57569 | 0.7800 | 0.7840 |
| likelihood_math_random_quality_gate | 0.57497 | 0.7800 | 0.7840 |
| likelihood_math_noaux | 0.56224 | 0.7880 | 0.7840 |
| positive_math_equation | 0.55677 | 0.7780 | 0.7820 |

## Interpretation limits

Primary evidence concerns equation-based grouping beyond its matched random control, not answer-equality clusters as error labels.
Outcome-supervised quality features are shared by independent and random controls. Positive single-only contrasts do not prove grouping value.
All intervals resample questions. Multiple arm contrasts are exploratory. Seed repeats use the same questions and are not pooled as independent samples.
Policy feasibility is empirical. Generated tokens exclude TF, text-feature and quality-head cost. No v3 end-to-end latency claim.
Asymptote inputs are text only. Unsupported arithmetic checks are missing, not a proof that a derivation is correct.

[Full metrics](./evaluation.json) · [Policy lock](./selection_lock.json)

## Observed math-check coverage

{'responses': 16000, 'numeric_check_fraction': 0.797124981880188, 'numeric_any_fail_fraction': 0.0886249989271164, 'constraint_check_fraction': 0.004000000189989805, 'constraint_any_fail_fraction': 0.0}

Quality AUROC/BCE are response-level descriptive measures; responses from one question are dependent.

## Training seed repeats

{'42': {'likelihood_math_quality_gate': {'mean_raw_nll': 0.5756888389587402, 'accuracy': 0.786, 'mean_tokens': 16446.548, 'mean_samples': 11.29, 'test_point_constraint_met': True}, 'likelihood_math_random_quality_gate': {'mean_raw_nll': 0.5749687552452087, 'accuracy': 0.786, 'mean_tokens': 16447.9, 'mean_samples': 11.3, 'test_point_constraint_met': True}, 'likelihood_math_independent': {'mean_raw_nll': 0.9463225603103638, 'accuracy': 0.788, 'mean_tokens': 11591.674, 'mean_samples': 7.86, 'test_point_constraint_met': True}}, '43': {'likelihood_math_quality_gate_seed43': {'mean_raw_nll': 0.5975257754325867, 'accuracy': 0.778, 'mean_tokens': 9674.968, 'mean_samples': 7.17, 'test_point_constraint_met': True}, 'likelihood_math_random_quality_gate_seed43': {'mean_raw_nll': 0.6004666686058044, 'accuracy': 0.776, 'mean_tokens': 8806.966, 'mean_samples': 6.44, 'test_point_constraint_met': True}, 'likelihood_math_independent_seed43': {'mean_raw_nll': 0.9740480780601501, 'accuracy': 0.792, 'mean_tokens': 11353.802, 'mean_samples': 7.712, 'test_point_constraint_met': True}}, '44': {'likelihood_math_quality_gate_seed44': {'mean_raw_nll': 0.5845301747322083, 'accuracy': 0.788, 'mean_tokens': 12498.038, 'mean_samples': 8.394, 'test_point_constraint_met': True}, 'likelihood_math_random_quality_gate_seed44': {'mean_raw_nll': 0.5836491584777832, 'accuracy': 0.788, 'mean_tokens': 12557.112, 'mean_samples': 8.438, 'test_point_constraint_met': True}, 'likelihood_math_independent_seed44': {'mean_raw_nll': 0.9790684580802917, 'accuracy': 0.788, 'mean_tokens': 11689.886, 'mean_samples': 7.89, 'test_point_constraint_met': True}}}

## Exploratory ablation NLL contrasts

{'equation_representation_positive': {'left': 'positive_equation', 'right': 'positive_raw', 'difference': -0.004716313364246162, 'ci95': [-0.010773491473329703, 0.0011187514103090472], 'questions': 500, 'repetitions': 5000}, 'equation_representation_likelihood': {'left': 'likelihood_math_equation', 'right': 'likelihood_math_raw', 'difference': 0.003614061283184128, 'ci95': [-8.066769090000884e-05, 0.007270709709661601], 'questions': 500, 'repetitions': 5000}, 'math_features_independent': {'left': 'likelihood_math_independent', 'right': 'likelihood_base_independent', 'difference': 0.2114518257527234, 'ci95': [0.12347016270438362, 0.31108428252132814], 'questions': 500, 'repetitions': 5000}, 'partition_discount': {'left': 'likelihood_math_equation', 'right': 'likelihood_math_independent', 'difference': -0.3670458884880372, 'ci95': [-0.4931067573950565, -0.2561985784834044], 'questions': 500, 'repetitions': 5000}, 'quality_gate': {'left': 'likelihood_math_quality_gate', 'right': 'likelihood_math_equation', 'difference': -0.0035879030439664346, 'ci95': [-0.007037775247578793, -0.00019370903784024632], 'questions': 500, 'repetitions': 5000}, 'auxiliary_supervision': {'left': 'likelihood_math_equation', 'right': 'likelihood_math_noaux', 'difference': 0.01704142168677754, 'ci95': [-0.004449740084039513, 0.04110574528560648], 'questions': 500, 'repetitions': 5000}, 'likelihood_vs_positive_score': {'left': 'likelihood_math_equation', 'right': 'positive_math_equation', 'difference': 0.022511621385530815, 'ci95': [-0.013367235883052444, 0.06052394069958558], 'questions': 500, 'repetitions': 5000}}

---

## Historical source: `artifacts/runs/gemma3_pgm_v3/result_summary.md`

# PGM v3 결과 요약

2026-11-01 15:22 KST 완료. 23개 head 학습, 새 calibration 450문제와 holdout 500문제의 30,400응답 생성·TF·baseline confidence 추출을 모두 마쳤다. 모델·보정 설정은 holdout 생성·평가 전에 고정했고, 완료 후 v1/v2/v3 artifact hash 검증도 통과했다.

## 결론

**이번 v3에서는 equation grouping의 추가 효과를 확인하지 못했고, primary PGM의 비용도 증가했다. 기존 v2를 대체할 근거가 부족하다.**

| 같은 v3 holdout의 방법 | Accuracy | 평균 응답 수 | 평균 생성 token |
|---|---:|---:|---:|
| V3 primary PGM | 78.6% | 11.290 | 16,446.5 |
| Matched random | 78.6% | 11.300 | 16,447.9 |
| Independent quality | 78.8% | 7.860 | 11,591.7 |
| ReASC | 77.8% | 3.450 | 4,176.0 |
| CGES | 77.4% | 6.430 | 8,019.5 |
| Frozen v2 | 77.6% | 2.888 | 3,365.5 |
| SC32 | 77.6% | 32.000 | 29,301.5 |

## Primary mechanism test

PGM−matched random raw mean NLL 차이는 +0.000720, 95% paired-bootstrap interval [-0.000688,+0.002165]였다. 0을 포함하며 grouping 우위를 뒷받침하지 않는다. 두 방법은 500문제의 최종 correct/incorrect 결과도 모두 같았다.

Observed-answer winner agreement는 고정 t=8에서 100%, t=32에서 99.6%였다. Equation/step proxy가 random grouping을 넘어 answer selection을 바꾸는 효과가 작았다.

## 비용과 해석

Frozen v2 대비 primary PGM은 accuracy +1.0%p, 생성 token 4.89배였다. Paired accuracy 차이의 95% interval은 [-0.8,+2.8]%p다. Point-estimate accuracy 증가를 확정적인 개선으로 해석하지 않는다.

Primary policy는 calibration-B에서 min=2, cap=32, gamma=0.8, T≈1.996으로 선택됐다. 초기 min=2, cap=8인 frozen v2와 비교해 장기 sampling 비용이 커졌다. Test를 보고 cap이나 threshold를 다시 선택하지 않았다.

V2의 이전 82.4%와 이번 78.6%는 다른 cohort의 값이다. 직접적인 before/after 차이로 해석하지 않고 위의 같은 holdout frozen v2 77.6%와 비교한다.

## Ablation과 제한

- Equation representation의 positive-score NLL 개선 interval은 0을 포함했다. Likelihood-score에서는 개선하지 못했다.
- Quality-aware gate는 geometry-only 대비 NLL을 조금 줄였지만 matched random도 같은 성능이어서 semantic grouping의 이득과 구분해야 한다.
- Auxiliary BCE와 negative likelihood score의 추가 효과는 확보되지 않았다. Math feature를 쓴 independent 모델의 raw NLL은 base-feature 모델보다 나빴다.
- Numeric check coverage는 79.7%였지만 question substitution coverage는 0.4%였고 failure flag가 한 번도 켜지지 않았다. 이 구성은 강한 question-grounded verifier가 아니다.
- 추가 seed 43/44에서도 PGM/random NLL 차이는 작았고 방향이 일관되지 않았다.
- Generated-token 결과는 replay이며 v3 online latency는 아직 측정하지 않았다.

[상세 보고서](./report.md) · [전체 지표](./evaluation.json) · [고정 정책](./selection_lock.json) · [정확도–비용 figure](./accuracy_tokens.png)

---

## Historical source: `artifacts/runs/gemma3_pgm_v4/control_audit.md`

# V4 Phase A: evidence controls

Spent development data only; no new responses.

| Gate / evidence | Raw mean NLL(2/4/8) | Accuracy@8 |
|---|---:|---:|
| frozen_real_gate/real | 0.338148 | 0.858 |
| frozen_real_gate/random | 0.338523 | 0.860 |
| frozen_real_gate/mass_only | 0.338428 | 0.858 |
| uniform6/real | 0.347371 | 0.858 |
| uniform6/random | 0.347327 | 0.858 |
| uniform6/mass_only | 0.347400 | 0.858 |

## Paired question contrasts

{'frozen_real_gate/real_minus_random': {'NLL': {'mean': -0.00037479808855257204, 'interval95': [-0.0017969198509435592, 0.0008279039728914324]}, 'prefixes': {'2': {'winner_agreement': 1.0, 'TV': 0.0, 'corrected': 0, 'regressed': 0}, '4': {'winner_agreement': 1.0, 'TV': 0.002014085718892461, 'corrected': 0, 'regressed': 0}, '8': {'winner_agreement': 0.998, 'TV': 0.003690064706165401, 'corrected': 0, 'regressed': 1}}}, 'frozen_real_gate/real_minus_mass_only': {'NLL': {'mean': -0.00028001817998427335, 'interval95': [-0.0015994391086628684, 0.0007749546003351205]}, 'prefixes': {'2': {'winner_agreement': 1.0, 'TV': 0.0, 'corrected': 0, 'regressed': 0}, '4': {'winner_agreement': 1.0, 'TV': 0.001867306891457464, 'corrected': 0, 'regressed': 0}, '8': {'winner_agreement': 0.996, 'TV': 0.0031389750688297275, 'corrected': 1, 'regressed': 1}}}, 'uniform6/real_minus_random': {'NLL': {'mean': 4.3049647994694904e-05, 'interval95': [-0.0020290549943209105, 0.002144884462004585]}, 'prefixes': {'2': {'winner_agreement': 1.0, 'TV': 0.0, 'corrected': 0, 'regressed': 0}, '4': {'winner_agreement': 1.0, 'TV': 0.0031996237081091668, 'corrected': 0, 'regressed': 0}, '8': {'winner_agreement': 0.992, 'TV': 0.005983749114785588, 'corrected': 1, 'regressed': 1}}}, 'uniform6/real_minus_mass_only': {'NLL': {'mean': -2.9100744653743072e-05, 'interval95': [-0.0018925898586111187, 0.0016155675922710033]}, 'prefixes': {'2': {'winner_agreement': 1.0, 'TV': 0.0, 'corrected': 0, 'regressed': 0}, '4': {'winner_agreement': 1.0, 'TV': 0.0030599024131495495, 'corrected': 0, 'regressed': 0}, '8': {'winner_agreement': 0.994, 'TV': 0.005125017816024891, 'corrected': 1, 'regressed': 1}}}}

Original v3 forward reproduced, max absolute logp difference: 3.3417545317604436e-06

Frozen v3 global temperature is used for every calibrated counterfactual. No new temperature was selected.
Residual pair statistics are descriptive and do not label a shared reasoning error.

---

## Historical source: `artifacts/runs/gemma3_pgm_v4/development_report.md`

# V4 development results

**Development evidence only; no fresh or previous holdout labels were opened.**

| Seed | Arm | Raw NLL(2/4/8) | Accuracy@8 | Selected settings |
|---|---|---:|---:|---|
| 42 | I0 | 0.532407 | 0.852 | {} |
| 42 | I1 | 0.417249 | 0.852 | {} |
| 42 | S0 | 0.390706 | 0.850 | {'alpha': 0.5} |
| 42 | H0 | 0.412424 | 0.852 | {'rho': 0.5} |
| 42 | H2 | 0.430886 | 0.854 | {'rho': 0.25} |
| 42 | H4 | 0.430886 | 0.854 | {'rho': 0.25, 'slot': 0} |
| 42 | H1 | 0.412864 | 0.852 | {'rho': 0.5} |
| 42 | H5 | 0.412149 | 0.852 | {'rho': 0.5} |
| 42 | H3 | 0.415907 | 0.848 | {'rho': 0.5} |
| 43 | I0 | 0.532833 | 0.850 | {} |
| 43 | I1 | 0.419825 | 0.850 | {} |
| 43 | S0 | 0.393681 | 0.848 | {'alpha': 0.5} |
| 43 | H0 | 0.420230 | 0.850 | {'rho': 0.5} |
| 43 | H2 | 0.432124 | 0.856 | {'rho': 0.25} |
| 43 | H4 | 0.432124 | 0.856 | {'rho': 0.25, 'slot': 0} |
| 43 | H1 | 0.420734 | 0.852 | {'rho': 0.5} |
| 43 | H5 | 0.419904 | 0.852 | {'rho': 0.5} |
| 43 | H3 | 0.433719 | 0.848 | {'rho': 0.5} |
| 44 | I0 | 0.537331 | 0.850 | {} |
| 44 | I1 | 0.420576 | 0.850 | {} |
| 44 | S0 | 0.393500 | 0.850 | {'alpha': 0.5} |
| 44 | H0 | 0.424128 | 0.850 | {'rho': 0.5} |
| 44 | H2 | 0.435551 | 0.854 | {'rho': 0.25} |
| 44 | H4 | 0.435551 | 0.854 | {'rho': 0.25, 'slot': 0} |
| 44 | H1 | 0.423984 | 0.848 | {'rho': 0.5} |
| 44 | H5 | 0.424017 | 0.850 | {'rho': 0.5} |
| 44 | H3 | 0.441415 | 0.850 | {'rho': 0.25} |

Primary seed: 42. Strongest control selected on seed-42 dev: H3.

Development gate: FAIL

{'primary_nll_gain_random': False, 'primary_nll_gain_control': False, 'accuracy8_vs_random': True, 'accuracy8_vs_control': True, 'consistent_seeds_at_least2': True, 'audits_passed': True}

Global quality probability is held fixed across aggregation arms. The shared OTHER head is also fixed.
Train q is OOF from one held-out model, while dev q averages five models; their variance differs.
rho/alpha/slot are selected on this spent dev cohort. These results are not confirmatory generalization estimates.

[Control audit](./control_audit.md) · [Gate](./development_gate.json)

---

## Historical source: `artifacts/runs/gemma3_pgm_v4/quality_oof_report.md`

# V4 question-OOF quality audit

Each fold has 760 training and 190 held-out questions. Scalers were fitted only to the 760 training questions.
Quality uses BCE only, 100 fixed epochs per fold. All aggregation arms receive the same probabilities.

| Seed | Train OOF BCE | Dev ensemble BCE | Dev AUROC (descriptive) | Between-fold probability std |
|---|---:|---:|---:|---:|
| 42 | 0.23713 | 0.30094 | 0.89923 | 0.01114 |
| 43 | 0.23700 | 0.30058 | 0.89933 | 0.01037 |
| 44 | 0.23577 | 0.30064 | 0.89915 | 0.00971 |

BCE intervals and reliability bins are saved in each quality/seed*/report.json.
BCE uncertainty resamples questions. AUROC is a descriptive response-level metric.
OOF train predictions come from one held-out model; dev uses five-model probability averages. Their variance differs.

---

## Historical source: `artifacts/runs/gemma3_pgm_v4/report.md`

# PGM v4 결과 요약

2026-11-02 03:17 KST 완료. **사전 지정 development gate 실패로 이번 v4 실험을 종료했다.**

기존 train 950 / dev 500 bank를 재사용하여 9개 aggregation arm을 seeds 42/43/44로 비교했다. Quality 15 fits, OTHER 3 fits, generic scale 3 fits로 총 21개 optimization run을 완료했다. 새 응답 생성은 0회이며 fresh calibration/holdout은 생성하거나 평가하지 않았다.

## Primary seed 42 — spent dev 500

| Method | Raw NLL(2/4/8) | Accuracy@8 |
|---|---:|---:|
| Shared-reliability PGM | 0.412424 | 85.2% |
| Matched random | 0.412864 | 85.2% |
| Answer-only grouping | 0.415907 | 84.8% |
| Selected single | 0.430886 | 85.4% |
| Independent | 0.532407 | 85.2% |
| Independent + scale | 0.417249 | 85.2% |
| Size discount | 0.390706 | 85.0% |

## 판정

실제 grouping의 NLL 개선은 random 대비 0.000440 nats, seed-42 dev에서 선택한 strongest non-reasoning control(H3) 대비 0.003483 nats였다. 둘 다 사전 진입 기준 0.005 nats에 미달했다.
Accuracy 조건과 2/3-seed 방향 조건은 통과했지만, 조건을 모두 만족해야 하므로 gate는 실패다. 기준을 사후 변경하지 않았다.

PGM−random raw NLL의 seed별 차이: 42: -0.000440, 43: -0.000505, 44: +0.000143.
Seed 42의 PGM−random descriptive 95% interval: [-0.002972428004205268, 0.0016331171122721392].
이 interval은 같은 dev에서 rho를 선택한 뒤의 descriptive 분석이며 confirmatory evidence가 아니다.

## 해석

Independent aggregation에 generic scale을 추가하면 NLL이 크게 줄었다. Shared reliability도 independent control보다 낫지만, 실제 reasoning grouping과 matched random의 차이는 작고 seed별 방향도 일관되지 않았다.
새 latent model은 개별 response quality를 그대로 유지한다. 따라서 이번 실험에서는 quality 변화와 dependency modeling의 효과를 분리했지만, reasoning grouping의 충분한 추가 가치는 확보하지 못했다.
기존 size discount S0는 이 dev에서 가장 낮은 NLL을 보였다. 이것은 spent dev에서의 선택 결과이며 fresh 성능이나 PGM-specific superiority의 증거로 승격하지 않는다.
위 85%대 accuracy는 dev cohort의 fixed-budget 결과다. V2/V3 holdout accuracy나 adaptive policy token cost와 직접 비교하지 않는다. Operational reference는 frozen v2를 유지한다.

## 검증

전체 78 tests 통과. 기존 v1/v2/v3 및 v4 development artifact hash 검증 통과. 15개 fold의 question overlap=0과 training-only scaler를 checkpoint에서 재검증했다. 모든 selected prediction의 normalization과 finite 값을 확인했다.
Phase A에서 원래 v3 forward를 active-candidate logp 오차 3.35e-6 이내로 재현했다. 기존 v3 real grouping과 mass-only control의 dev NLL 차이 interval도 0을 포함했다.

![Development NLL](./development_nll.png)

[전체 development 결과](./development_report.md) · [Control audit](./control_audit.md) · [OOF quality audit](./quality_oof_report.md) · [Gate](./development_gate.json) · [Paired 분석](./completion_analysis.json) · [검증](./validation.json)

# V4 development results

**Development evidence only; no fresh or previous holdout labels were opened.**

| Seed | Arm | Raw NLL(2/4/8) | Accuracy@8 | Selected settings |
|---|---|---:|---:|---|
| 42 | I0 | 0.532407 | 0.852 | {} |
| 42 | I1 | 0.417249 | 0.852 | {} |
| 42 | S0 | 0.390706 | 0.850 | {'alpha': 0.5} |
| 42 | H0 | 0.412424 | 0.852 | {'rho': 0.5} |
| 42 | H2 | 0.430886 | 0.854 | {'rho': 0.25} |
| 42 | H4 | 0.430886 | 0.854 | {'rho': 0.25, 'slot': 0} |
| 42 | H1 | 0.412864 | 0.852 | {'rho': 0.5} |
| 42 | H5 | 0.412149 | 0.852 | {'rho': 0.5} |
| 42 | H3 | 0.415907 | 0.848 | {'rho': 0.5} |
| 43 | I0 | 0.532833 | 0.850 | {} |
| 43 | I1 | 0.419825 | 0.850 | {} |
| 43 | S0 | 0.393681 | 0.848 | {'alpha': 0.5} |
| 43 | H0 | 0.420230 | 0.850 | {'rho': 0.5} |
| 43 | H2 | 0.432124 | 0.856 | {'rho': 0.25} |
| 43 | H4 | 0.432124 | 0.856 | {'rho': 0.25, 'slot': 0} |
| 43 | H1 | 0.420734 | 0.852 | {'rho': 0.5} |
| 43 | H5 | 0.419904 | 0.852 | {'rho': 0.5} |
| 43 | H3 | 0.433719 | 0.848 | {'rho': 0.5} |
| 44 | I0 | 0.537331 | 0.850 | {} |
| 44 | I1 | 0.420576 | 0.850 | {} |
| 44 | S0 | 0.393500 | 0.850 | {'alpha': 0.5} |
| 44 | H0 | 0.424128 | 0.850 | {'rho': 0.5} |
| 44 | H2 | 0.435551 | 0.854 | {'rho': 0.25} |
| 44 | H4 | 0.435551 | 0.854 | {'rho': 0.25, 'slot': 0} |
| 44 | H1 | 0.423984 | 0.848 | {'rho': 0.5} |
| 44 | H5 | 0.424017 | 0.850 | {'rho': 0.5} |
| 44 | H3 | 0.441415 | 0.850 | {'rho': 0.25} |

Primary seed: 42. Strongest control selected on seed-42 dev: H3.

Development gate: FAIL

{'primary_nll_gain_random': False, 'primary_nll_gain_control': False, 'accuracy8_vs_random': True, 'accuracy8_vs_control': True, 'consistent_seeds_at_least2': True, 'audits_passed': True}

Global quality probability is held fixed across aggregation arms. The shared OTHER head is also fixed.
Train q is OOF from one held-out model, while dev q averages five models; their variance differs.
rho/alpha/slot are selected on this spent dev cohort. These results are not confirmatory generalization estimates.

[Control audit](./control_audit.md) · [Gate](./development_gate.json)

---

## Historical source: `artifacts/runs/gemma3_pgm_v4/result_summary.md`

# PGM v4 결과 요약

2026-11-02 03:17 KST 완료. **사전 지정 development gate 실패로 이번 v4 실험을 종료했다.**

기존 train 950 / dev 500 bank를 재사용하여 9개 aggregation arm을 seeds 42/43/44로 비교했다. Quality 15 fits, OTHER 3 fits, generic scale 3 fits로 총 21개 optimization run을 완료했다. 새 응답 생성은 0회이며 fresh calibration/holdout은 생성하거나 평가하지 않았다.

## Primary seed 42 — spent dev 500

| Method | Raw NLL(2/4/8) | Accuracy@8 |
|---|---:|---:|
| Shared-reliability PGM | 0.412424 | 85.2% |
| Matched random | 0.412864 | 85.2% |
| Answer-only grouping | 0.415907 | 84.8% |
| Selected single | 0.430886 | 85.4% |
| Independent | 0.532407 | 85.2% |
| Independent + scale | 0.417249 | 85.2% |
| Size discount | 0.390706 | 85.0% |

## 판정

실제 grouping의 NLL 개선은 random 대비 0.000440 nats, seed-42 dev에서 선택한 strongest non-reasoning control(H3) 대비 0.003483 nats였다. 둘 다 사전 진입 기준 0.005 nats에 미달했다.
Accuracy 조건과 2/3-seed 방향 조건은 통과했지만, 조건을 모두 만족해야 하므로 gate는 실패다. 기준을 사후 변경하지 않았다.

PGM−random raw NLL의 seed별 차이: 42: -0.000440, 43: -0.000505, 44: +0.000143.
Seed 42의 PGM−random descriptive 95% interval: [-0.002972428004205268, 0.0016331171122721392].
이 interval은 같은 dev에서 rho를 선택한 뒤의 descriptive 분석이며 confirmatory evidence가 아니다.

## 해석

Independent aggregation에 generic scale을 추가하면 NLL이 크게 줄었다. Shared reliability도 independent control보다 낫지만, 실제 reasoning grouping과 matched random의 차이는 작고 seed별 방향도 일관되지 않았다.
새 latent model은 개별 response quality를 그대로 유지한다. 따라서 이번 실험에서는 quality 변화와 dependency modeling의 효과를 분리했지만, reasoning grouping의 충분한 추가 가치는 확보하지 못했다.
기존 size discount S0는 이 dev에서 가장 낮은 NLL을 보였다. 이것은 spent dev에서의 선택 결과이며 fresh 성능이나 PGM-specific superiority의 증거로 승격하지 않는다.
위 85%대 accuracy는 dev cohort의 fixed-budget 결과다. V2/V3 holdout accuracy나 adaptive policy token cost와 직접 비교하지 않는다. Operational reference는 frozen v2를 유지한다.

## 검증

전체 78 tests 통과. 기존 v1/v2/v3 및 v4 development artifact hash 검증 통과. 15개 fold의 question overlap=0과 training-only scaler를 checkpoint에서 재검증했다. 모든 selected prediction의 normalization과 finite 값을 확인했다.
Phase A에서 원래 v3 forward를 active-candidate logp 오차 3.35e-6 이내로 재현했다. 기존 v3 real grouping과 mass-only control의 dev NLL 차이 interval도 0을 포함했다.

![Development NLL](./development_nll.png)

[전체 development 결과](./development_report.md) · [Control audit](./control_audit.md) · [OOF quality audit](./quality_oof_report.md) · [Gate](./development_gate.json) · [Paired 분석](./completion_analysis.json) · [검증](./validation.json)


## Refactor provenance

- Historical selector `selection_lock.json` protocol hash: `735343526451643bbb0babc5b4846b78ea07fe45d1323ccdadc8b8f1607c7319`. This predates the path/code refactor.
- Historical state `selection_lock.json` protocol hash: `11b661521a7ed6fb96ca921dcb6829ea97a41b5e51f6a09c1fbfbc0c118a28ea`. This predates the path/code refactor.
- Historical state `model_lock.json` protocol hash: `02c4ff1d3334e2379d7dd859f8eea86f54be812fc4e99fbbb5faceed925d18dc`. This predates the path/code refactor.
- Historical state `completion_lock.json` protocol hash: `63137e587383c791b4788d1925cf50a72ae47025b5653ce5b51eeff00bc71eaa`. This predates the path/code refactor.
- Historical state `review_lock.json` protocol hash: `4ee461ba4552d7bf551089054947724d7f3253acabefa3491e17b821f9d754b9`. This predates the path/code refactor.


## Removed V1-checkpoint control on the historical V2 holdout

V2의 새500문제에서 이전 V1 checkpoint와 calibration을 그대로 적용한 domain-shift control이다. 현재 active selector evaluation에서는 제거했고, 당시 수치는 이 Markdown에 보존한다.

| Control | Accuracy | Mean responses | Mean generated tokens |
|---|---:|---:|---:|
| Original V1 frozen mixture on V2 holdout500 | 0.826000 | 7.310000 | 11622.598000 |
