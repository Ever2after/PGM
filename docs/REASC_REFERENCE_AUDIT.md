# ReASC reference implementation audit

작성: 2026-11-02 KST. 범위: 제공된 archive와 공식 저장소를 읽고 비교한 static audit. 모델 실행·설치·학습·성능 측정은 수행하지 않았다.

## 1. 확인한 원본

| Source | Local path | Identity |
|---|---|---|
| 사용자 archive | `pgm_reasc.zip` | SHA256 `b4427aa382baf221df0a9d241c025241d46de121fee74d6cacc3181987c0ccab` |
| Archive 추출본 | `third_party/reasc_reference/` | 21 files; `__MACOSX`, `.DS_Store`, AppleDouble metadata 제외; 코드 내용 변경 없음 |
| 공식 GitHub | `third_party/reasc_official/` | `junseokkim00/ReASC`, commit `8ee1c76ea0c04bb3cdb81d09bc2655dbc3a8cecb` |
| 공식 commit 시각 | — | 2026-04-15 14:03:15 +09:00; commit message `first commit` |

공식 permalink: https://github.com/junseokkim00/ReASC/tree/8ee1c76ea0c04bb3cdb81d09bc2655dbc3a8cecb

Archive README는 ReASC(confsc)를 PGM으로 대체할 예정이므로 제외했다고 설명한다. Archive의 SC/ASC/ESC, 생성·calibration·answer extraction·judge 공통 파일 11개와 공식 repo의 대응 파일은 byte-identical이다. MATH train/test와 Omni-Math 데이터 3개도 byte-identical이다. 공식 repo의 `confsc_offline_code.ipynb`, `confsc_online_code.ipynb`가 빠진 ReASC 본체를 제공한다. Archive의 `generate_samples_with_confidence.py`는 추가 PGM용 코드이며 공식 repo에는 없다.

이는 제공된 구현 사이의 일치 여부다. 공식 repo에 공개된 version/revision 이외에 저자의 실제 논문 실행 환경·모델 revision·실험 bank까지 동일하다고 확인한 것은 아니다.

## 2. 실제 생성 설정

| Item | Code behavior | Evidence |
|---|---|---|
| Main inference | vLLM, model의 `get_default_sampling_params()` | `sc_with_self_certainty.py`, `utils_hf.py` |
| Bank budget | 질문당 16 responses | runner의 `--batch_size 16`; notebook의 prefix 1..16 |
| Maximum output | MATH/GSM8K/GPQA 1,024; Omni-Math 2,048 | test 생성기·runner |
| Seed | 0 | runner, Python/NumPy/Torch seed 설정 |
| Math prompt | user content에 question 그대로 입력 | 추가 step-by-step/boxed instruction 없음 |
| MCQA prompt | answer-choice instruction 추가 | `build_prompt` 및 test 생성기 |
| GPU use | visible GPU 전체에 tensor parallel | `tensor_parallel_size=torch.cuda.device_count()` |
| Confidence request | `SamplingParams.logprobs=20` | sampling top-k와 별개 |
| Gemma sampling | 현지 pinned model config의 top-p=.95, top-k=64; temperature는 model config에 없으므로 resolved vLLM default를 기록해야 함 | `generation_config.json`; 논문은 temperature=1.0 |
| Framework | Python3.10.16, vLLM0.11.0, Torch2.8.0, Transformers4.56.2 | 공식 `environment.yaml` |
| Numerical libraries | NumPy2.2.6, SciPy1.15.3, scikit-learn1.7.2 | 같은 environment file |

현재 우리 vLLM0.18.0과 TF4.57.6 환경은 위 환경과 다르다. 동일한 scalar sampling setting만으로 sample token까지 동일하다고 주장할 수 없다. vLLM config·tokenizer·모델 revision·EOS·request order·parallelism을 함께 기록해야 한다.

## 3. ReASC confidence: 논문 설명·우리 기존 구현과의 차이

공식 생성기는 반환된 각 token의 logprob dictionary D에 대해 다음 proxy를 저장한다.

$$
c_i^{code}=-\log K_i-\frac{1}{K_i}\sum_{v\in D_i}\log p_i(v),\qquad K_i=|D_i|.
$$

- full-vocabulary score가 아니라 요청한 top-20 dictionary 기반이다. vLLM이 sampled token을 추가로 반환하면 K가20보다 클 수 있으므로 실제 dictionary를 보존한다.
- sampled token의 logprob, dictionary entropy, full-vocabulary entropy는 서로 다른 신호다.
- window128을 stride1로 만들지만 **마지막 suffix의 짧은 window도 포함**한다. 길이 T이면 window도 T개다.
- Bottom10은 `max(1,T//10)`개 window 평균의 평균이다. 우리 기존 구현의 full-length-window·ceil 방식과 다르다.
- Offline notebook cell7은 **lowest-group confidence**로 mean/std/gate를 fit하지만, cell8·16·17의 inference score는 **Bottom10group confidence**다. 이 불일치를 코드 재현에서는 유지하고, 일관된 Bottom10 calibration은 별도 sensitivity로 표시한다.
- Online notebook은 test question별 첫 response의 Bottom10 score로 mean/std 및 2-component full-covariance GMM을 fit한다. `random_state=0`, target=.90. `calibration_size=128` 변수는 있으나 fitting을128개로 제한하지 않는다.
- Gate 외에도 weighted Beta threshold가 첫 response에서 충족되면 t=1에 stop할 수 있다. 공통 minimum2를 baseline에 강제하면 원본 ReASC가 아니다.

ReASC notebook permalink:

- [Offline](https://github.com/junseokkim00/ReASC/blob/8ee1c76ea0c04bb3cdb81d09bc2655dbc3a8cecb/confsc_offline_code.ipynb)
- [Online](https://github.com/junseokkim00/ReASC/blob/8ee1c76ea0c04bb3cdb81d09bc2655dbc3a8cecb/confsc_online_code.ipynb)

## 4. Stop rule와 baseline 구현

- Evidence update는 `max(1, exp(.7*z))`이고 top-two weights a,b에 대한 probability는 `Beta(a+1,b+1)`의 upper-half mass다. Threshold=.95, cap16.
- Offline gate는 empirical accepted accuracy target=.90를 처음 만족하는 threshold와 correct-confidence mean의 최댓값이다. 목표를 만족하지 못할 때 notebook의 best-observed fallback도 parity 대상으로 기록한다. 이 empirical gate는 population accuracy guarantee가 아니다.
- Online은 높은 mean의 GMM component를 correct surrogate로 간주하고, observed score 중 posterior≥.90의 첫 threshold와 높은 component mean의 최댓값을 사용한다. 교차점이 없으면 observed maximum으로 fallback한다.
- ASC는 .95 top-two Beta rule이다. `asc_code_dirichlet.ipynb`는 해당 snapshot에서 핵심 계산이 여전히 top-two Beta이므로 독립적인 Dirichlet baseline으로 세지 않는다.
- ESC는 disjoint4-window unanimity가 stop trigger이지만, 최종 answer는 **그때까지 누적한 전체 prefix의 majority**다. 기존 우리 ESC의 unanimous-window answer 선택과 구분해야 한다.
- 정답 문자열 tie는 첫 등장 answer가 우선이다.

## 5. Answer extraction·correctness·cost

1. 생성 prompt에는 boxed instruction이 없으며, 생성 뒤 같은 LLM에 추가 user turn을 보내 answer를 추출한다. Extraction max_tokens=128이고 default stochastic sampling을 사용한다.
2. `sc_with_self_certainty.py`에서 이미 extraction을 실행한 뒤 shell runner는 `compute_prediction.py`로 다시 extraction을 수행하여 pred를 덮어쓴다. Inline extraction은 이후 question의 sampling RNG history에도 영향을 줄 수 있다. 두 호출을 임의로 하나로 합치면 엄밀한 token-level 재현은 아니다.
3. `compute_prediction.py`의 batch_size1 branch는 questions를 list 대신 string으로 넘기는 문제가 있어, runnable adapter에서 이 interface bug를 고치고 기본 batch128 branch와 비교한다.
4. `utils_hf.py`의 HF token과 xVerify path는 placeholder다. 실제 token을 source에 쓰지 않고 기존 인증 경로를 사용하며, judge model·revision을 manifest에 고정해야 한다.
5. xVerify는 extracted answer가 아닌 **원래 response 전체**를 판정한다. Notebook은 같은 pred string에 대응하는 response verdict의 majority로 candidate correctness를 만들고, True/False tie는 True로 처리한다. ESC의 verdict 집계는 전체16개를 사용한다. 이것은 selected final answer를 직접 판정하는 metric과 동일하지 않다. Verdict를 answer 선택이나 stopping input에 사용하지는 않는다.
6. Notebook cost는 consumed prefix의 **prompt+assistant를 다시 chat-template로 tokenization한 길이**에 rounded parameter count N을 곱한다. 논문의 2N FLOPs/token 근사와 coefficient가 다르며, answer extraction·TF feature pass·PGM fitting·GMM overhead는 포함하지 않는다. 원본 proxy와 end-to-end cost를 구분한다.
7. Archive PGM generator는 sampled-token logprob을 token ID로 찾지 않고 rank0 또는 argmax fallback으로 찾는다. Stochastic decoding에서는 다른 token의 logprob을 저장할 수 있다. 추가 구현에서는 `seq.token_ids[t]`를 dictionary key로 사용하고 exact-ID teacher forcing으로 검증한다. Reference ReASC proxy는 원래 dictionary 식을 유지한다.
8. Append-only JSONL에는 자동 중복 방지·resume·fingerprint 검사가 없다. 신규 collector는 response identity를 넣고 duplicate append를 방지해야 한다.

## 6. 실제 데이터 규모·출처

| File | Rows | Normalized unique problems | Audit |
|---|---:|---:|---|
| `math500/train.jsonl` |12,000|11,998|unique_id 기준 train7,499행/test4,501행. 원본 train 전용 파일이 아님 |
| `math500/test.jsonl` |500|500|기존 모든 PGM train/dev/calibration cohort와 overlap0; 과거에는 이미 평가했던 benchmark |
| `omnimath/test.jsonl` |1,000|999|exact duplicate1개. Question text로 group하면 notebook 평가 단위가999개가 됨 |

Archive·공식 데이터 hash:

- MATH development pool: `90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076`
- MATH500 test: `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`
- Omni-Math test: `eee05021a27a889fedba7023d4fa59d8a930f7eb51b3c08b6cc4e1cd959d3109`

기존 exclusion registry를 적용하면 development pool에 미사용 unique7,324개가 있다. 그중 original-train ID pool은6,174개, original-test ID pool은1,150개다. Calibration128을 먼저 고정한 뒤 PGM용 original-train800/dev200, fresh original-test500을 확보할 수 있다. 실제 선택은 normalized problem identity와 original HF split을 교차 검증한다.

## 7. 구현 시 재현 원칙

`reference_code`와 `corrected_evaluation`을 별도 named profile로 유지한다. 먼저 공식 notebook의 선택 answer·stop time·proxy cost를 동일 synthetic bank에서 재현한다. Shared bank·sampling·extraction은 동일하게 유지하고, correctness·cost metric의 정의 차이만 별도로 보고한다. 논문 설명에 맞춘 confidence 변경은 추가 arm이며, 이를 공식 코드의 원본 성능으로 대신 제시하지 않는다.

이 audit는 기존 Frozen-v2/V5 lock·checkpoint·실험 결과를 수정하지 않는다. 상세 구현·실험 순서는 [comparison plan](./REASC_PGM_COMPARISON_PLAN.md)에 있다.
