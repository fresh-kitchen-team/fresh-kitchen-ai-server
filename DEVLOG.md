# fresh-kitchen-ai-server — DEVLOG

> 개발 과정에서의 의사결정·문제 해결·모델 진화 기록

> 냉장고 식재료 관리 앱 **fresh-kitchen** 의 AI 백엔드 서버 구축 프로젝트.
> 한 장의 사진으로 **음식 분류 · 영수증 OCR · 냉장고 인벤토리 자동 구축** 을 제공하는 3종 ML 파이프라인을 FastAPI 단일 서버에 통합했습니다.

---

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| **프로젝트명** | fresh-kitchen-ai-server |
| **기간** | 2026.03 ~ 2026.05 (3개월, 산학협력 프로젝트) |
| **역할** | AI 백엔드 **단독 개발** (모델 학습 · 데이터 파이프라인 · API 서버) |
| **팀 구성** | 모바일 앱 (2) · 백엔드 (3) · AI (본인) |
| **결과물** | 70-class 한국 식재료 분류 모델 (val_acc **96.60%**) + 3개 엔드포인트 운영 서버 |

---

## 핵심 성과

| 지표 | 값 |
|---|---|
| **분류 정확도** | val_acc **96.60%** (70 클래스, 한국 식재료) |
| **클래스 수 확장** | ver2: 35개 → ver5: **70개** (2배) |
| **모델 진화** | 4회 재학습으로 +2.04%p 누적 개선 |
| **응답 시간** | EfficientNet 직접 분류 ~150ms / Gemini 폴백 ~3s |
| **자동 데이터 수집** | Gemini 폴백 시 `auto_labeled/` 에 자동 누적 (재학습 데이터 자가 축적) |
| **운영 안정성** | 토큰 인증 · 매직바이트 이중 검증 · 타임아웃 제어 · 500 오류 비노출 |

---

## 아키텍처

```
┌─────────────────┐                  ┌──────────────────────────────────────┐
│  fresh-kitchen  │   HTTPS + Bearer │       fresh-kitchen-ai-server        │
│   (모바일 앱)    │ ───────────────► │            (FastAPI)                 │
└─────────────────┘                  │                                      │
                                     │  ┌────────────────────────────────┐  │
                                     │  │ POST /food-classification      │──┼─► EfficientNet V2-M
                                     │  │                                │  │   ↓ confidence < 80%
                                     │  │                                │  │   Gemini 2.5 Flash (폴백)
                                     │  ├────────────────────────────────┤  │
                                     │  │ POST /receipt-ocr              │──┼─► Document AI → Gemini 2.5 Flash
                                     │  ├────────────────────────────────┤  │
                                     │  │ POST /fridge-detection         │──┼─► Gemini 2.5 Flash Vision
                                     │  └────────────────────────────────┘  │
                                     └──────────────────────────────────────┘
```

---

## 개발 타임라인

> Git 이력 이전(3~4월)은 당시 기록·자문 피드백 메모를 기반으로 재구성, 5월 이후는 커밋·학습 로그 기준.

| 시기 | 단계 | 내용 |
|---|---|---|
| **3.16** | 분류기 착수 | EfficientNet 백본 freeze 후 head 학습으로 첫 분류기 구성 시작 |
| **4.02** | 전처리 정비 | 크롤링 데이터와 실사 사진의 분포 차이를 메우기 위한 증강 추가 (`RandomPerspective`·`GaussianBlur`·`ColorJitter` 등) |
| **4.03** | 1차 자문 피드백 | EfficientNet 단일 분류 + YOLO 냉장고 전체 인식 구도 검토. 32 클래스 E0에서 ~98%, "100 클래스까지는 V2-M" 권고. 유통기한은 UI 입력, 냉장고는 YOLOv8 계획 |
| **4.05** | 백본 교체 | 32-class EfficientNet-E0(~98%) → 클래스 확장 대비해 **V2-M** 으로 전환 |
| **4.16** | 영수증 OCR 2단계 구현 | Document AI로 1차 텍스트 추출 → Gemini API로 2차(구매일·식재료) 추출 구조 구현 |
| **4.23** | 파인튜닝 고도화 | 클래스별 가중치 저장 + backbone 상위 4개 블록(`features[-4:]`) 해동 |
| **4.24** | 2차 자문 피드백 | 평가용 reference dataset 필요성 제기 + "그냥 VLM 쓰는 건 어때?" 제안. 이를 계기로 **VLM을 Auto-Labeler로 쓰는 Self-improving 4단계 구조를 직접 설계** |
| **4.27** | VLM 결합 | 자문 피드백 반영해 분류기 + VLM 폴백 구조 결합 검토 |
| **5.02~** | Git 이력 시작 / 3파이프라인 연동 | 서버·파이프라인 코드 형상관리 시작. 3종 파이프라인을 FastAPI로 통합하며 **백엔드와 JSON 응답 형식을 통일**해가며 수정 |
| **5.13** | ver3 / 냉장고 전환 | 60 클래스 재학습(val_acc 94.73%). 냉장고 감지를 **YOLO → Gemini Vision** 으로 확정 (COCO 한국 식재료 미포함 + 연계안 다중 호출 지연 + YOLO-World 한국어 불안정) |
| **5.27** | ver4 → ver5 ⭐ | 70 한글 클래스 + Progressive Unfreezing(ver4, 95.42%) → Early Stopping patience 완화로 backbone 학습량 확보(ver5, val_acc **96.60%**) |
| **5.29** | 운영 안정화 | 타임아웃·인증·문서 정비 |

---

## 주요 기술적 의사결정

### 1. 백본 모델 선택 — EfficientNet-E0 → V2-M

**문제**: 초기에는 EfficientNet-E0 으로 32 클래스 분류기를 만들었음. 32 클래스에서는 정확도가 나쁘지 않았으나, 최종 목표는 70+ 클래스 확장이라 "클래스 수가 늘어나도 성능이 유지될까?" 가 풀리지 않았음.

**해결**: 백본을 EfficientNet **V2-M** 으로 교체. 같은 EfficientNet 계열이지만 V2 는 학습 안정성·표현력이 한 단계 위라 클래스 확장 여지가 충분.

**효과**: 35 → 60 → 70 클래스로 단계적으로 확장하는 동안 val_acc 94~96% 대를 유지. 이후 모든 진화(데이터 확장·하이퍼파라미터 튜닝)의 기반이 됨.

### 2. Progressive Unfreezing 파인튜닝

**진화 과정**: 학습 가능 범위를 점진적으로 풀면서 4 단계에 걸쳐 정착.

1. **초기 — 가중치 전체 학습**: 200~500장 규모 데이터에 비해 학습 파라미터가 너무 많아 과적합·표현 붕괴.
2. **1차 — backbone 전체 freeze, head 만 학습**: 안정적이지만 한국 식재료 도메인 특화가 약함.
3. **2차 — `features[-4:]` 마지막 4개 블록 + head 해동**: 표현력은 향상됐으나 catastrophic forgetting 우려.
4. **현재 — Progressive Unfreezing** (`training/train_EfficientNet_V2_M.py`):
   - Epoch 5 까지: head + backbone 상위 4개 블록(`features[-4:]`)만 학습 (LR=1e-4)
   - Epoch 5 부터: backbone 전체 해동 + 새로 풀린 하위 레이어는 별도 param_group 으로 LR=1e-5 (10배 작게)

**효과**: 사전학습 표현을 보존하면서 도메인 특화. 같은 데이터에서 head-only 학습 대비 안정적으로 수렴.

### 3. 신뢰도 기반 자동 폴백 (Self-improving 시스템)

**문제**: 70개 클래스 분류기는 학습 데이터에 없는 식재료(예: 수박, 망고)에 대해 강제로 잘못된 라벨을 반환함.

> **계기는 4/24 자문의 "그냥 VLM 쓰는 건 어때?" 한마디.** 자문은 VLM 활용을 가볍게 제안한 정도였고, 거기서 한 발 더 나아가 ① 분류 → ② 저신뢰 시 VLM 위임 → ③ VLM 라벨 자동 축적 → ④ 다음 학습에 반영하는 **self-improving 4단계 구조는 직접 설계**했음.

**해결**:
- EfficientNet softmax 최댓값이 **80% 미만**이면 Gemini 2.5 Flash Vision으로 자동 위임
- Gemini는 클래스 목록을 프롬프트로 받아 **목록에 있으면 해당 클래스명**, 없으면 **자유 한국어 라벨**을 반환
- 폴백된 이미지는 `dataset/auto_labeled/{label}/` 에 자동 저장 → **다음 학습 라운드의 데이터셋이 자가 누적**되는 self-improving 구조

**효과**: 모델 한계를 외부 LLM으로 즉시 보완 + 학습 데이터 수집 자동화. 운영하면서 데이터가 늘어남.

### 4. 2단계 OCR — Document AI + Gemini JSON 강제

**문제**: Document AI는 텍스트 추출은 잘하지만 **"구매일 추출"** 과 **"식재료만 골라내기"** 같은 의미 기반 작업을 못함.

**해결**:
1. **1단계** — Document AI 로 영수증의 `line_item` 엔티티만 정규식 `[^가-힣\s]` 으로 한글 필터링
2. **2단계** — 추출된 품목 목록과 전체 OCR 텍스트를 Gemini 2.5 Flash 에 전달, `response_mime_type="application/json"` 로 **JSON 강제** 응답 받음
3. Gemini가 `purchasedAt` (YYYY-MM-DD) 와 식재료만 필터된 `ingredients[]` 를 반환
4. 정규식으로 날짜 형식 재검증 → 신뢰성 확보

**효과**: 브랜드명 자동 제거 (`"CJ 비비고 만두"` → `"만두"`), 카드명·환불·할인·합계 등 비식재료 완전 제외.

### 5. 냉장고 감지 — YOLO 대신 Gemini Vision 채택

**초기 계획**: YOLOv8n (COCO 80 클래스) 으로 냉장고 전체 인식.

**검토 과정**:
1. **COCO 클래스 한계** — 된장·고추장·두부 등 한국 식재료 미포함 → 실질 탐지 불가
2. **YOLO + EfficientNet 연계안** — YOLO로 위치 탐지 후 crop → EfficientNet 분류. 그러나 확신도 80% 미만 시 Gemini 폴백 구조상 냉장고 내 다수 아이템에서 **Gemini 다중 호출 → 응답 지연**
3. **YOLO-World** — 텍스트로 클래스 지정은 가능하나 한국어 식재료 인식률 불안정

**최종 결정**: Gemini 2.5 Flash Vision 단독 사용 (5/13).
- 사진 1장 → 1회 호출 → 식재료 목록 JSON 반환, 호출 최소화로 응답 속도 확보
- 한국 식재료 인식률 가장 높고, 학습 데이터 0장으로 운영 시작
- 영수증 OCR과 동일한 `response_mime_type="application/json"` 패턴 재사용

LLM 특유의 동의어 중복·포괄어 출력은 프롬프트로 제어 (아래 "마주친 문제" 4·5번 참고).

### 6. 외부 API 타임아웃 제어

**문제**: Gemini API 특성상 응답 지연이 발생할 수 있고, 응답이 돌아오지 않으면 서버가 무한 대기(hang) 상태가 될 수 있음. 실제로 60초 초과 타임아웃이 발생한 적은 없으나, 운영 중 예외 상황에 대비한 안전장치가 필요.

**1차 해결 (ThreadPoolExecutor)**: 초기에는 각 모듈에 싱글톤 `ThreadPoolExecutor(max_workers=1)` 를 두고 `future.result(timeout=GEMINI_TIMEOUT)` 로 상한선을 걸었음. 하지만 두 가지 결함이 있었음.
- `future.cancel()` 은 **이미 실행 중인 스레드를 멈추지 못함** → 타임아웃 후에도 Gemini 호출이 백그라운드에서 유일한 워커를 계속 점유.
- `max_workers=1` 이라 점유된 워커가 풀릴 때까지 **다음 요청이 큐에서 대기** → 동시 요청 시 사용자 대기시간이 `GEMINI_TIMEOUT` 을 초과할 수 있음.

**현재 해결 (SDK 네이티브 타임아웃)**: `google-genai` SDK가 지원하는 `http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT * 1000)` (단위 ms) 로 전환. 타임아웃 시 **HTTP 요청 자체가 중단**되어 스레드/워커 점유가 없고, 스레드 풀 없이 코드도 단순해짐. 초과 시 기존 예외 처리로 빈 결과 반환 (재시도 없음).

### 7. 하드웨어 자동 감지

```python
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Apple Silicon MPS → NVIDIA CUDA → CPU 순으로 자동 선택. Mixed Precision(AMP) 은 CUDA 에서만 활성화 (MPS는 GradScaler 불안정).

### 8. 학습 환경 분리 — MacBook(MPS) → Windows(CUDA)

**문제**: 개발은 MacBook(Apple Silicon, MPS) 에서 진행했으나, 막상 학습을 돌려보니 두 가지 한계가 걸림.
- 480×480 입력 + EfficientNet V2-M + 배치 16 조합에서 **RAM 부족으로 swap thrashing** 발생 → 학습이 사실상 멈춤
- MPS 는 AMP(Mixed Precision) 미지원 → 1 epoch 당 수십 분, 50 epoch 학습이 현실적으로 불가능

**해결**: 학습 단계만 **집의 Windows 데스크탑(NVIDIA GPU)** 으로 분리. `#7` 의 자동 감지 로직 덕분에 동일 코드를 그대로 실행 → CUDA 환경에서 AMP 자동 활성화. 학습 산출물(`.pth`, `docs/logs/training_log_*.csv`) 만 운영 서버로 옮겨 반영.

**효과**: ver4·ver5 학습이 각각 수 시간 내 완료. **개발(Mac/MPS) ↔ 학습(Windows/CUDA) ↔ 운영(서버)** 3-tier 분리가 자연스럽게 정착.

### 9. 보안 설계

| 항목 | 구현 |
|---|---|
| 인증 | HTTP Bearer 토큰 + `hmac.compare_digest()` (타이밍 공격 방어) |
| 파일 검증 | Content-Type 헤더 + **실제 파일 magic bytes** 이중 검증 (헤더 위조 방어) |
| 토큰 누락 | 서버 시작 자체를 거부 (`RuntimeError`) |
| 임시 파일 | `tempfile.NamedTemporaryFile` + `finally` 블록 `os.unlink()` 즉시 삭제 |
| 500 응답 | 클라이언트에는 고정 메시지만, 스택 트레이스는 `server.log` 에만 |

---

## 모델 진화 과정

같은 EfficientNet V2-M 백본으로 데이터·하이퍼파라미터를 점진 개선하며 4회 재학습.

| 버전 | 클래스 수 | 데이터셋 | 최대 Epoch | Patience | 최고 val_acc | 주요 변경 |
|---|---|---|---|---|---|---|
| **ver2** | 35 (영문) | Train 7,315 / Val 1,803 | 30 | - | 94.56% | 초기 베이스라인 |
| **ver3** | 60 (영문) | Train 11,050 / Val 2,731 | 30 | - | 94.73% | 클래스 +25 확장 |
| **ver4** | 70 (한글) | Train 12,714 / Val 3,147 | 30 | 5 | 95.42% | 한글 클래스명 전환 + Progressive Unfreezing 도입 |
| **ver5** ⭐ | 70 (한글) | Train 12,714 / Val 3,147 | **50** | **8** | **96.60%** | Early Stopping patience 완화 |

### ver4 → ver5 의 결정적 차이

**ver4의 문제**: 학습 로그(`docs/logs/training_log_5_27.csv`) 분석 결과, backbone 전체 해동이 epoch 6에 일어났고 epoch 7에 바로 최고점(95.42%) 달성 → **저장된 best 모델 기준 backbone은 단 2 에폭(epoch 6·7)만 학습된 상태.** 이후 patience=5 소진으로 epoch 12에 조기 종료되지만, 저장된 가중치는 epoch 7 것이라 backbone이 충분히 수렴하지 못함.

**ver5의 해결**: `EPOCHS=50, PATIENCE=8` 로 완화하여 backbone 충분한 학습 시간 확보. ReduceLROnPlateau가 LR을 자동 반감(1e-5 → 5e-6 → ... → 6.25e-7)하며 epoch 29에 최고점 달성 → epoch 37에 조기 종료.

### ver5 클래스별 정확도 개선 (vs ver4)

**크게 향상된 클래스** (시각적 다양성이 큰 클래스):

| 클래스 | ver4 | ver5 | 변화 |
|---|---|---|---|
| 베이컨 | 69.2% | 88.5% | **+19.3%p** |
| 버터 | 84.8% | 95.7% | +10.9%p |
| 무 | 90.0% | 100.0% | +10.0%p |
| 치즈 | 61.0% | 70.7% | +9.7%p |
| 소금 | 90.2% | 97.6% | +7.4%p |

**100% 완벽 분류 클래스 수**: ver4 19개 → ver5 **24개** (+5)

---

## 마주친 문제와 해결

### 1. Gemini 2.5 Flash 응답 포맷 변경 대응

Gemini 2.5 가 단일 객체를 종종 `[{...}]` 배열로 감싸 반환하면서 운영 중 `'list' object has no attribute 'get'` 에러 발생.

```python
result = json.loads(response.text)
if isinstance(result, list):
    result = result[0] if result else {}
if not isinstance(result, dict):
    return {"error": "Gemini 응답 형식 오류"}
```

→ 방어 로직 추가로 외부 API 변경에도 안정 동작.

### 2. 신뢰도 임계값 trade-off

70 클래스로 늘어나면서 softmax 최댓값이 자연 감소 → 임계값 80에 잦은 폴백 발생.
임계값 70 으로 낮춰봤으나 **77% 로 잘못 분류하는 케이스** 확인 → 다시 80으로 복원.

**교훈**: 임계값 하향보다 **모델 자체 개선(ver5)이 정공법**. 클래스 수 증가 시 백본 학습량 확보가 필수.

### 3. 냉장고 감지 프롬프트 — 포괄적 식품명 출력

초기 프롬프트는 `"즉석식품"`, `"음식"` 같은 추상 카테고리를 식재료로 반환.

**해결**: 프롬프트에 명시적 제외 규칙 추가.
> "즉석식품", "음식", "채소", "소스", "식품" 처럼 포괄적인 단어는 쓰지 말고, 정확히 식별 안 되면 해당 항목을 제외해줘.

### 4. 동의어 중복 — "계란" vs "달걀", "콜라" vs "탄산음료"

같은 식재료를 두 가지 이름으로 반환하는 문제 → 프롬프트에 통합 규칙 추가.
> 같은 식재료는 중복 없이 하나만 써줘 (예: "계란"과 "달걀"은 "계란" 하나로).

### 5. 데이터 증강 — 크롤링 이미지와 실제 스마트폰 사진의 분포 차이

**문제**: 학습 데이터는 DuckDuckGo로 크롤링한 이미지 — 정면·단독·고화질이 대부분. 반면 실제 앱 사용자는 스마트폰으로 찍은 **흐릿하거나 각도가 틀어진** 사진을 올림. 같은 모델인데 크롤링 데이터 정확도보다 실제 사진 정확도가 낮게 나오는 문제.

**해결**: 학습 전처리에 스마트폰 촬영 조건을 시뮬레이션하는 증강 추가 (4/2, `training/train_EfficientNet_V2_M.py`).

| 증강 | 목적 |
|---|---|
| `RandomPerspective(distortion_scale=0.2)` | 비스듬한 촬영 각도 |
| `GaussianBlur(sigma=(0.1, 2.0))` | 초점 흐림 |
| `ColorJitter(brightness=0.3, ...)` | 조명 변화 |
| `RandomAdjustSharpness` | 선명도 편차 |
| `RandomRotation(30)` | 회전된 사진 |

**효과**: 크롤링 데이터만으로 학습했지만 실제 스마트폰 촬영 사진에서도 안정적인 추론 가능. train-test 분포 차이를 데이터 없이 증강으로 보정.

### 6. 카테고리 값 정규화 — LLM 자유 출력 방어

세 파이프라인 모두 Gemini가 카테고리(`VEGETABLE`·`MEAT` 등)를 직접 반환하는데, 대소문자 혼용·오타·정의에 없는 값(`FOOD`, `기타` 등)을 내놓는 경우가 있었음.

**해결**: 공통 모듈 `models/category.py` 의 `normalize_category()` 로 일원화 — 허용 집합(`VALID_CATEGORIES`) 에 없으면 무조건 `ETC` 로 보정. 백엔드로 나가는 카테고리 값이 항상 enum 범위 안에 있도록 보장.

---

## 데이터 파이프라인

새 클래스 추가 시 5단계 자동 파이프라인:

1. **`data_crawl.py`** — DuckDuckGo (`ddgs` 패키지) 로 클래스별 이미지 자동 다운로드
2. **`data_clean.py`** — MD5 해시로 중복 제거 (split 간 누수 차단)
3. **`data_split.py`** — train → val 분리 (실행 시 val 초기화로 멱등성 보장)
4. **`data_diet.py`** — 클래스별 이미지 수 상한 (train ≤ 400, val ≤ 80) — 불균형 완화
5. **`data_len.py`** — 클래스별 통계 출력

학습 시점에는 추가로 `remove_corrupted_images()` 가 PIL로 모든 이미지를 로드해 손상 파일 자동 제거 + `WeightedRandomSampler` 로 클래스 불균형 보정.

---

## 회고 및 배운 점

### 잘한 점

- **외부 LLM을 분류기 보조로 사용** 하는 self-improving 구조 설계 — 모델의 한계를 운영으로 자가 보완.
- **학습 로그 기반 의사결정** — ver4가 "최고점이 epoch 7" 인 걸 보고 backbone 학습량 부족 진단 → ver5 patience 완화로 +1.18%p 개선.
- **외부 API 변경 대응 패턴** — Gemini 응답 포맷 변경에도 isinstance 가드로 안전.

### 아쉬운 점 / 개선 여지

- **치즈 70.7%** — 여전히 전체 최하위. 시각적 다양성(슬라이스/크림/모짜렐라 등) 때문. 별도 sub-class 분리 또는 hard mining 필요.
- **돼지고기·오징어 회귀** — ver5에서 일부 클래스가 ver4 대비 -10%p 가까이 떨어짐. 시각적 유사 클래스(소고기·새우)와의 confusion 분석 후 hard negative 보강 필요.
- **학습 인프라 외부 의존** — 학습은 Windows 데스크탑(CUDA) 으로 분리해 해결했으나, 팀 차원의 공용 학습 환경이 없어 재현·인수인계 시 마찰. 클라우드 GPU(Colab Pro·Lambda 등) 표준화가 다음 과제.
- **실사 기반 test set 부재** — val_acc 96.60% 는 크롤링 이미지(깨끗한 정면·고화질) 기준 수치. 실제 앱에서 스마트폰으로 찍은 사진에 대한 정량 평가를 못해 운영 성능과 수치 사이의 갭이 얼마인지 확인되지 않음. 실사 이미지로 구성된 별도 test set 확보가 필요.

### 배운 점

- **Early Stopping patience 는 모델 복잡도·클래스 수에 비례** 해야 한다. ver4에서 patience=5 는 35→70 클래스 확장 후 너무 짧았음.
- **80% 신뢰도 임계값은 70 클래스 분류기의 "노이즈 영역" 경계** 였음. 그 미만은 차라리 외부 LLM에 위임하는 게 정확도·UX 모두에 유리.
- **외부 LLM은 결정적이지 않다** — JSON 강제(`response_mime_type`), 타임아웃, isinstance 가드, 정규식 재검증 같은 다중 안전장치 필수.
- **`hmac.compare_digest`, magic bytes 검증** 같은 보안 기본기를 직접 구현하면서 "왜 이런 게 필요한지" 체득.
- **학습 데이터와 실제 배포 환경의 분포 차이는 증강으로 좁힐 수 있다** — 크롤링 데이터(정면·고화질)로만 학습하면 스마트폰 실사 사진에서 성능이 떨어진다. 별도 실사 데이터 없이도 RandomPerspective·GaussianBlur 같은 증강 몇 줄로 갭을 상당히 줄일 수 있다는 걸 직접 확인.

---

## 기술 스택

| 분류 | 사용 기술 |
|---|---|
| **언어** | Python 3.10.19 |
| **웹 프레임워크** | FastAPI 0.136, uvicorn 0.46 |
| **딥러닝** | PyTorch 2.11, torchvision 0.26 |
| **모델** | EfficientNet V2-M (timm/torchvision) |
| **외부 API** | Google Cloud Document AI, Google Gemini 2.5 Flash (`google-genai`) |
| **이미지** | Pillow 12.2 |
| **데이터 수집** | ddgs (DuckDuckGo 크롤러) |
| **기타** | python-dotenv, hmac, tempfile |

---

## 링크

- **레포지토리**: https://github.com/fresh-kitchen-team/fresh-kitchen-ai-server
- **상세 개발 계획서**: `docs/상세개발계획서 AI담당.xlsx`
- **학습 로그 데이터**: `docs/logs/training_log_*.csv` (ver2~ver5 전체 보존)
