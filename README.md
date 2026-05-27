# fresh-kitchen-ai-server

> 냉장고 식재료 관리 앱 **fresh-kitchen** 의 AI 백엔드 서버.
> 한 장의 사진으로 **음식 분류 · 영수증 인식 · 냉장고 인벤토리 자동 구축** 을 수행합니다.

[![Python](https://img.shields.io/badge/Python-3.10.19-blue)]() [![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688)]() [![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C)]()

---

## 한눈에 보는 시스템

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

### 핵심 파이프라인 3개

| 엔드포인트 | 입력 | 출력 | 사용 모델 |
|---|---|---|---|
| `/internal/v1/food-classification` | 식재료 사진 1장 | 분류 결과 + Top3 + 카테고리 | EfficientNet V2-M (70 클래스) + Gemini 폴백 |
| `/internal/v1/receipt-ocr` | 영수증 사진 1장 | 구매일 + 식재료 목록 + 카테고리 | Google Document AI + Gemini 2.5 Flash |
| `/internal/v1/fridge-detection` | 냉장고 내부 사진 1장 | 식재료 목록 + 카테고리 | Gemini 2.5 Flash Vision |

### 설계 포인트

- **신뢰도 기반 자동 폴백** — 분류기가 80% 미만으로 확신하면 Gemini가 대신 판단하고, 결과 이미지를 `dataset/auto_labeled/`에 자동 저장 (self-improving 학습 데이터 축적).
- **외부 API 통합** — Gemini는 단순 LLM이 아니라 **Vision + JSON 강제 모드** 로 사용. `response_mime_type="application/json"` 으로 파싱 안정성 확보.
- **하드웨어 자동 감지** — Apple Silicon MPS → NVIDIA CUDA → CPU 순. Mixed Precision(AMP)은 CUDA에서만 활성화.

---

## 빠른 시작

### 1. 사전 요구사항

- Python **3.10.19**
- macOS / Linux / Windows
- (선택) NVIDIA GPU — Mixed Precision 학습용
- Google Cloud 계정 — Document AI 활성화
- Gemini API 키

### 2. 설치

```bash
# 저장소 클론
git clone <repo-url> fresh-kitchen-ai-server
cd fresh-kitchen-ai-server

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 편집기로 열어 API 키 입력 (아래 [환경 변수] 참고)

# 모델 가중치 배치 (팀 드라이브에서 다운로드)
# best_food_model_v2_m_ver4.pth → 프로젝트 루트
```

### 3. 서버 실행

```bash
python3 -m uvicorn main:app --reload --port 8000
```

| 항목 | 주소 / 경로 |
|---|---|
| Swagger UI | http://127.0.0.1:8000/docs |
| 서버 로그 | `./server.log` |
| 외부 노출 (선택) | `ngrok http 8000` |

### 4. 동작 확인 (curl)

```bash
# 환경 변수에서 토큰 읽어서 호출
TOKEN=$(grep AI_SECRET_TOKEN .env | cut -d= -f2)

curl -X POST http://127.0.0.1:8000/internal/v1/food-classification \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@picture_model/predict/beef1.jpeg"
```

---

## 환경 변수

`.env.example`을 복사한 뒤 모든 값을 채워주세요. **`.env` 파일은 절대 git에 커밋되지 않습니다** (gitignore 등록됨).

```env
# Google Cloud Document AI (영수증 OCR)
GOOGLE_APPLICATION_CREDENTIALS=./receipt_model/your-service-account-key.json
PROJECT_ID=your-gcp-project-id
PROCESSOR_ID=your-document-ai-processor-id
LOCATION=us

# Google Gemini API (음식 분류 폴백 · 영수증 2차 정제 · 냉장고 감지 공용)
GEMINI_API_KEY=your-gemini-api-key

# API 인증 토큰 (Bearer 방식, 백엔드 팀에 전달)
AI_SECRET_TOKEN=your-secret-token

# Gemini 호출 타임아웃 (초, 기본값 30 — 생략 가능)
GEMINI_TIMEOUT=30
```

| 변수 | 필수 | 설명 |
|---|:---:|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | ✓ | GCP 서비스 계정 키 경로 (상대 경로 OK — 코드가 절대경로로 변환) |
| `PROJECT_ID` / `PROCESSOR_ID` / `LOCATION` | ✓ | Document AI 프로세서 정보 |
| `GEMINI_API_KEY` | ✓ | Gemini API 키 (3개 파이프라인 공통) |
| `AI_SECRET_TOKEN` | ✓ | 백엔드 ↔ AI 서버 간 Bearer 토큰. 누락 시 서버가 시작을 거부합니다 |
| `GEMINI_TIMEOUT` |  | Gemini 호출 타임아웃 (기본 30초) |

**토큰 생성**:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## API 레퍼런스

### 공통 사항

#### 인증
모든 엔드포인트는 HTTP Bearer 토큰을 요구합니다.

```
Authorization: Bearer <AI_SECRET_TOKEN>
```

| 상황 | 상태 코드 |
|---|---|
| 토큰 없음·불일치 | `401` |
| 파일 크기 10MB 초과 | `413` |
| 허용되지 않는 형식 (jpg/png/webp 외) | `415` |
| 서버 내부 오류 | `500` |

#### 파일 검증
- 최대 크기: **10 MB**
- 허용 MIME 타입: `image/jpeg`, `image/png`, `image/webp`
- Content-Type 헤더와 **실제 파일 magic bytes** 두 단계로 검증 (헤더 위조 방어)

#### 카테고리 enum
세 엔드포인트 모두 응답의 `category` 필드에 아래 값 중 하나를 반환합니다.

| 값 | 의미 |
|---|---|
| `VEGETABLE` | 채소 |
| `FRUIT` | 과일 |
| `MEAT` | 육류 |
| `SEAFOOD` | 해산물 |
| `DAIRY` | 유제품 |
| `GRAIN` | 곡물 · 면류 |
| `SAUCE` | 소스 · 양념 |
| `DRINK` | 음료 |
| `ETC` | 기타 |

정규화 로직은 [`models/category.py`](models/category.py)에서 관리. Gemini가 비표준 값을 반환해도 자동으로 `ETC`로 변환됩니다.

---

### 1️⃣ 음식 분류 — `POST /internal/v1/food-classification`

식재료 사진 한 장에서 클래스명과 카테고리를 추출합니다.

**Request**
```
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: <식재료 이미지>
```

**Response (3가지 케이스)**

<details>
<summary><b>✅ EfficientNet 직접 분류 — confidence ≥ 80%</b></summary>

```json
{
  "bestMatch": "김치",
  "category": "VEGETABLE",
  "confidence": 98.42,
  "top3": [
    { "name": "김치",   "confidence": 98.42 },
    { "name": "고추장", "confidence":  1.13 },
    { "name": "된장",   "confidence":  0.31 }
  ],
  "source": "efficientnet"
}
```
</details>

<details>
<summary><b>🤖 Gemini 폴백 사용 — confidence < 80%, Gemini 성공</b></summary>

```json
{
  "bestMatch": "수박",
  "category": "FRUIT",
  "confidence": 54.31,
  "top3": [],
  "source": "gemini",
  "auto_saved": "dataset/auto_labeled/수박/auto_20260527_142315_123456.jpg"
}
```
- `top3`는 비어있음 (Gemini는 단일 라벨만 반환)
- 입력 이미지가 `dataset/auto_labeled/{label}/` 에 자동 저장됨
- **클래스 목록에 없는 식재료도 한국어로 자유롭게 반환** (예: 수박, 망고)
</details>

<details>
<summary><b>⚠️ Gemini도 실패 — EfficientNet 결과로 폴백</b></summary>

```json
{
  "bestMatch": "소고기",
  "category": "MEAT",
  "confidence": 42.10,
  "top3": [...],
  "source": "efficientnet_fallback"
}
```
</details>

#### `source` 필드 의미

| 값 | 상황 |
|---|---|
| `efficientnet` | 분류기가 80% 이상 확신 |
| `gemini` | 분류기 확신 부족 → Gemini 폴백 성공 (이미지 auto_labeled 저장) |
| `efficientnet_fallback` | Gemini 폴백도 실패 → 분류기 원래 결과 사용 |

---

### 2️⃣ 영수증 OCR — `POST /internal/v1/receipt-ocr`

영수증 사진에서 **구매일** 과 **식재료 목록** 을 추출합니다.

**Request**: 음식 분류와 동일한 multipart/form-data.

**Response**
```json
{
  "purchasedAt": "2026-05-13",
  "ingredients": [
    { "name": "두부",   "category": "ETC"     },
    { "name": "계란",   "category": "ETC"     },
    { "name": "만두",   "category": "GRAIN"   },
    { "name": "고등어", "category": "SEAFOOD" }
  ]
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `purchasedAt` | `string \| null` | `YYYY-MM-DD`. 인식 실패 시 `null` |
| `ingredients[].name` | `string` | 한글 식재료명 (브랜드 제거됨) |
| `ingredients[].category` | `string` | 카테고리 enum |

**참고**:
- Document AI가 글자를 전혀 찾지 못하면 Gemini 호출을 건너뛰고 `{"purchasedAt": null, "ingredients": []}` 반환.
- Gemini가 브랜드명을 자동 제거: `"CJ 비비고 만두"` → `"만두"`, `"풀무원 두부"` → `"두부"`
- 카드명·포인트·환불·할인·합계·부가세·봉투 등 **비식재료는 모두 제외** 됨.

---

### 3️⃣ 냉장고 식재료 감지 — `POST /internal/v1/fridge-detection`

냉장고 내부 사진에서 식재료 목록을 추출합니다.

**Response**
```json
{
  "items": [
    { "name": "계란",   "category": "ETC"       },
    { "name": "우유",   "category": "DAIRY"     },
    { "name": "두부",   "category": "ETC"       },
    { "name": "당근",   "category": "VEGETABLE" },
    { "name": "된장",   "category": "SAUCE"     }
  ]
}
```

**프롬프트 규칙**:
- 식재료·식품·음료만 포함
- 브랜드명 제거
- 한글 출력
- 뚜껑/포장재로 가려진 항목 제외
- 동의어 중복 제거 (예: "계란" ↔ "달걀" → 하나로)

Gemini 오류·타임아웃 시 `{"items": []}` 반환 (재시도 안 함 — 사용자 대기시간 최소화 정책).

---

## 시스템 동작 흐름

### 요청 처리 라이프사이클

```
1. 클라이언트 요청 도착
       ↓
2. Bearer 토큰 검증 (hmac.compare_digest — 타이밍 공격 방어)
       ↓
3. 파일 읽기 + 크기/MIME/매직바이트 검증
       ↓
4. OS 임시 파일로 저장 (tempfile.NamedTemporaryFile)
       ↓
5. 각 파이프라인 실행
       ↓
6. JSON 응답 반환
       ↓
7. finally 블록에서 임시 파일 즉시 삭제 (os.unlink)
```

### 음식 분류 내부 흐름

```
이미지 입력
    │
    ▼
torchvision transforms (Resize 512 → CenterCrop 480 → Normalize ImageNet)
    │
    ▼
EfficientNet V2-M 추론 → softmax × 100 → top-1 확신도
    │
    ├─ ≥ 80% ──────────────────► 즉시 반환 (source: "efficientnet")
    │
    └─ < 80%
            │
            ▼
        이미지 1024px 이하로 LANCZOS 리사이즈 → JPEG 85% 재인코딩
            │
            ▼
        Gemini 2.5 Flash 호출 (temperature=0, JSON 강제, 타임아웃 30초)
            │
            ├─ 성공 ──► auto_labeled/{label}/ 자동 저장 → 반환 (source: "gemini")
            │
            └─ 실패 ──► EfficientNet 결과 반환 (source: "efficientnet_fallback")
```

### 영수증 OCR 내부 흐름

```
영수증 이미지
    │
    ▼
[1단계] Google Document AI
    │   - entity.type_ == "line_item" 순회
    │   - 정규식 [^가-힣\s] 로 한글만 추출 (1자 이하 제거, 중복 제거)
    │   - 전체 OCR 텍스트도 함께 추출 (날짜 검색용)
    ▼
[2단계] Gemini 2.5 Flash
    │   - 입력: 1단계 품목 목록 + 전체 OCR 텍스트
    │   - 출력: purchasedAt (YYYY-MM-DD), ingredients[{name, category}]
    │   - JSON 강제 + 정규식으로 날짜 형식 재검증
    ▼
응답 반환
```

### 클라이언트(앱) 측 활용 예시

```javascript
// 사용자가 영수증 사진 업로드 → AI 서버에 전달
const form = new FormData();
form.append('file', receiptImage);

const res = await fetch(`${AI_SERVER}/internal/v1/receipt-ocr`, {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${AI_SECRET_TOKEN}` },
  body: form,
});

const { purchasedAt, ingredients } = await res.json();
// → DB에 저장하여 사용자의 식재료 인벤토리 구축
```

---

## 모델 학습 (Food Classifier)

**스크립트**: [`training/train_EfficientNet_V2_M.py`](training/train_EfficientNet_V2_M.py)

```bash
# dataset/train/, dataset/val/ 가 준비되어 있어야 합니다
python training/train_EfficientNet_V2_M.py
```

### 하이퍼파라미터

| 항목 | 값 |
|---|---|
| 기반 모델 | EfficientNet V2-M (ImageNet 사전학습) |
| 입력 크기 | 480 × 480 |
| 배치 크기 | 16 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-3) |
| 손실 함수 | CrossEntropy + Label Smoothing=0.1 |
| Gradient Clipping | max_norm=1.0 |
| LR Scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=3) |
| 최대 Epoch | 30 |
| Early Stopping | val_acc 기준 patience=5 |
| Dropout (분류기 헤드) | 0.3 |
| DataLoader workers | 4 |
| Mixed Precision (AMP) | CUDA에서만 활성화 (`torch.amp.GradScaler`) |

### 파인튜닝 전략 — Progressive Unfreezing

1. ImageNet 사전학습 가중치로 초기화 (`EfficientNet_V2_M_Weights.DEFAULT`)
2. **전체 freeze**
3. **`model.features[-4:]` 마지막 4개 블록 + 분류기 헤드만 해동** → LR `1e-4` 로 학습
4. **Epoch 5 이후 전체 레이어 해동** → backbone은 별도 param_group으로 LR `1e-5` (10배 작게)

### 데이터 증강 (Train만)

스마트폰 촬영 환경(원근 왜곡, 다양한 조명, 흔들림)을 모사:

```python
RandomResizedCrop(480, scale=(0.7, 1.0))
RandomHorizontalFlip()
RandomRotation(30)
RandomPerspective(distortion_scale=0.2, p=0.5)
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
RandomAdjustSharpness(sharpness_factor=2, p=0.3)
RandomGrayscale(p=0.05)
```

Val은 추론과 동일한 결정적 전처리(Resize 512 → CenterCrop 480).

### 클래스 불균형 처리

`WeightedRandomSampler` 사용. 각 클래스 샘플 수에 반비례하는 가중치(`1/count`)를 부여해 적은 클래스가 배치에 자주 포함됨.

### 손상 이미지 자동 제거

학습 직전 `remove_corrupted_images()` 가 `dataset/{train,val}` 의 모든 이미지를 PIL로 로딩하고, `IOError` 또는 `SyntaxError` 발생 파일을 자동 삭제.

### 체크포인트 형식

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,  # 추론 시 클래스 복원에 필수
}, SAVE_PATH)
```

`class_names`가 체크포인트에 포함되어 있어 **추론 시 `dataset/` 폴더 없이도** 클래스 매핑 복원 가능.

### 학습 로그

학습 종료 후 두 개의 CSV가 `docs/` 에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `training_log_5_27.csv` | 에폭별 train/val loss/acc, lr_head, lr_backbone |
| `training_log_class_acc_5_27.csv` | 베스트 모델 기준 클래스별 correct/total/accuracy |

### 단독 평가

```bash
python models/food_classifier/test_V2_M.py
```

`dataset/test/` 기준 전체 정확도와 클래스별 정확도를 출력. 폴더 클래스 순서가 체크포인트와 다르면 즉시 중단.

---

## 학습 이력

> 아래 ver2 표는 영문 클래스명을 쓰던 구버전입니다. **현재 운영 모델은 ver4 (한글 클래스명, 70개 클래스)**.

### ver4 — `docs/training_log_5_27.csv` (현재 운영본)
| 항목 | 값 |
|---|---|
| 학습 환경 | NVIDIA GPU (CUDA, AMP 활성화) |
| 클래스 수 | 70개 (한글) |
| 변경점 | 데이터 증강 강화, AMP 도입, 라면→라면사리·소시지→햄 등 클래스명 정비 |

### ver3 — `docs/training_log_5_13.csv`
| 항목 | 값 |
|---|---|
| 학습 환경 | Windows / NVIDIA GPU (CUDA) |
| 클래스 수 | 60개 (한글) |
| 데이터셋 | Train 11,050 / Val 2,731 |
| 학습 완료 | 22 Epoch (Early Stopping) |
| 최고 val_acc | **94.73%** (Epoch 17) |

### ver2 — `docs/training_log_5_3.csv`
| 항목 | 값 |
|---|---|
| 학습 환경 | Windows / NVIDIA GPU (CUDA) |
| 클래스 수 | 35개 (영문) |
| 데이터셋 | Train 7,315 / Val 1,803 |
| 학습 완료 | 30 Epoch (Early Stopping 미발동) |
| 최고 val_acc | **94.56%** (Epoch 27) |

---

## 데이터 파이프라인

새 클래스 데이터를 추가할 때 아래 순서로 실행:

```bash
# 1. 이미지 크롤링 — DuckDuckGo (ddgs 패키지)
python scripts/data_crawl.py

# 2. MD5 해시로 중복 이미지 제거 (dataset 전체 대상, split 간 누수도 차단)
python scripts/data_clean.py

# 3. train → val 분리 (실행 시 val 폴더 초기화로 멱등성 보장)
python scripts/data_split.py

# 4. 클래스별 이미지 수 상한 조정 (train ≤ 400장, val ≤ 80장)
python scripts/data_diet.py

# 5. 최종 클래스별 이미지 수 통계 출력
python scripts/data_len.py
```

### 데이터셋 구조

```
dataset/
├── train/<class>/*.jpg     # 학습 데이터
├── val/<class>/*.jpg       # 검증 데이터
├── test/<class>/*.jpg      # 테스트 데이터 (별도 평가용)
├── test_real_image/        # 영수증 OCR 단독 테스트용 실사 이미지
├── crawldata/              # 크롤링 원본 (정리 전)
└── auto_labeled/<label>/   # Gemini 자동 레이블링 결과 (재학습용 누적 데이터)
```

`dataset/` 전체는 `.gitignore` 에 등록되어 git에서 제외.

---

## 운영 / 디버깅

### 로깅

- 출력 형식: `%(asctime)s [%(levelname)s] %(name)s - %(message)s`
- 두 곳에 동시 기록: **콘솔(stdout)** + `./server.log` (UTF-8)
- 500 오류 발생 시 클라이언트에는 `"서버 내부 오류가 발생했습니다."` 만 노출, **스택 트레이스는 `server.log` 에만 기록**

### 모델 로딩

- 서버 부팅 시 FastAPI `lifespan` 훅에서 `load_food_model()` 1회 실행 → 메모리 상주
- Gemini · Document AI 클라이언트는 모듈 임포트 시점에 싱글톤 초기화

### 동시성 모델

- FastAPI 비동기 핸들러로 요청 처리
- Gemini 호출은 모듈별 `ThreadPoolExecutor(max_workers=1)` + `future.result(timeout=GEMINI_TIMEOUT)` 으로 타임아웃 제어
- PyTorch 추론은 `model.eval() + no_grad()` 컨텍스트에서 안전

### 단독 실행 (서버 없이 테스트)

각 모듈은 `__main__` 블록이 있어 단독으로 돌릴 수 있습니다:

```bash
python models/food_classifier/predict_V2_M.py     # picture_model/predict/beef1.jpeg 사용
python models/object_detection/fridge_detection.py # picture_model/predict/fridge_test.jpeg 사용
python models/receipt_ocr/receipt_ocr.py          # dataset/test_real_image/receipt.png 사용
```

---

## 보안

### 절대 커밋 금지

| 파일 | 이유 |
|---|---|
| `.env` | `GEMINI_API_KEY`, `AI_SECRET_TOKEN`, GCP 프로젝트 정보 |
| `receipt-app-*.json` | Google Cloud 서비스 계정 개인 키 |
| `receipt_model/` 폴더 | 서비스 계정 키 보관 위치 (폴더째 제외) |
| `*.pth` / `*.pt` | 모델 가중치 (200MB+) |
| `dataset/` | 학습 이미지 전체 |
| `picture_model/` | 로컬 테스트 이미지 |

위 항목은 모두 `.gitignore` 에 등록되어 있습니다.

### 인증

- HTTP **Bearer 토큰** 단일 방식
- 검증 시 `hmac.compare_digest()` 사용 → **타이밍 공격 방어**
- `AI_SECRET_TOKEN` 누락 시 서버 시작 자체를 거부 (`RuntimeError`)

### 파일 업로드

- **이중 검증**: Content-Type 헤더 + 실제 파일 magic bytes
- 10 MB 초과 시 즉시 거부 (413)
- 처리 직후 `os.unlink()` 로 임시 파일 즉시 삭제

### 토큰 유출 시 절차

1. `python3 -c "import secrets; print(secrets.token_hex(32))"` 로 새 토큰 생성
2. `.env` 의 `AI_SECRET_TOKEN` 교체
3. 서버 재시작
4. 백엔드 팀에 새 토큰 전달

### 내부 오류 비노출

500 응답은 항상 고정 메시지만 반환. 상세 예외는 `server.log` 에 `exc_info=True` 로 기록되어 디버깅 가능.

---

## 트러블슈팅

<details>
<summary><b>서버 시작 시 <code>RuntimeError: AI_SECRET_TOKEN이 .env에 설정되지 않았습니다</code></b></summary>

`.env` 파일에 `AI_SECRET_TOKEN` 값이 누락됐습니다. `python3 -c "import secrets; print(secrets.token_hex(32))"` 로 생성한 값을 `.env` 에 추가하세요.
</details>

<details>
<summary><b>모델 로딩 실패 — <code>FileNotFoundError: best_food_model_v2_m_ver4.pth</code></b></summary>

가중치 파일을 프로젝트 루트에 놓아야 합니다. 팀 드라이브에서 다운로드 후 `fresh-kitchen-ai-server/best_food_model_v2_m_ver4.pth` 경로에 저장하세요.
</details>

<details>
<summary><b>Document AI 호출 실패 — <code>403 PermissionDenied</code></b></summary>

1. `GOOGLE_APPLICATION_CREDENTIALS` 가 올바른 서비스 계정 키를 가리키는지 확인
2. 해당 서비스 계정에 **Document AI API 사용 권한**이 부여되어 있는지 확인
3. `PROJECT_ID`, `PROCESSOR_ID`, `LOCATION` 값이 GCP 콘솔의 프로세서 정보와 일치하는지 확인
</details>

<details>
<summary><b>Gemini 호출이 자꾸 타임아웃</b></summary>

- `.env` 에 `GEMINI_TIMEOUT=60` 처럼 더 큰 값을 설정해 보세요
- 이미지가 매우 크다면 사전에 클라이언트에서 리사이즈 권장 (서버도 1024px로 축소하긴 함)
- Gemini API quota 초과 가능성도 점검
</details>

<details>
<summary><b><code>data_crawl.py</code> 실행 시 <code>ModuleNotFoundError: No module named 'ddgs'</code></b></summary>

`pip install -r requirements.txt` 가 정상 완료됐는지 확인하세요. 이전 버전에서는 `duckduckgo_search` 였으나 현재는 `ddgs` 로 변경됐습니다.
</details>

<details>
<summary><b>학습 중 <code>OutOfMemoryError</code> (CUDA/MPS)</b></summary>

`training/train_EfficientNet_V2_M.py:18` 의 `BATCH_SIZE` 를 줄이세요 (16 → 8 → 4). MPS 환경에서 불안정하면 `NUM_WORKERS` 도 `0` 으로 낮추는 것을 권장.
</details>

<details>
<summary><b>음식 분류가 자주 <code>source: "gemini"</code> 로 떨어짐 (확신도 부족)</b></summary>

- 확신도 임계값 조정: `models/food_classifier/predict_V2_M.py:23` 의 `CONFIDENCE_THRESHOLD` 변경
- 또는 `dataset/auto_labeled/` 에 축적된 이미지를 학습 데이터에 편입하여 재학습
- 클래스 추가가 필요하면 `CLASS_CATEGORY` 딕셔너리에도 함께 추가 (누락 시 부팅 시 경고 로그 출력)
</details>

---

## 프로젝트 구조

<details>
<summary><b>전체 디렉토리 트리 (펼치기)</b></summary>

```
fresh-kitchen-ai-server/
├── main.py                                  # FastAPI 진입점, 3개 엔드포인트
├── requirements.txt                         # 직접 의존성 11개만 (transitive는 pip가 해결)
├── .env.example                             # 환경 변수 템플릿
├── .env                                     # 실제 API 키 (git 제외)
├── .gitignore
├── README.md
├── CLAUDE.md                                # AI 코딩 어시스턴트 작업 규칙
├── best_food_model_v2_m_ver4.pth            # 운영 모델 (git 제외, ~200MB)
│
├── models/
│   ├── category.py                          # 카테고리 enum + normalize_category()
│   ├── food_classifier/
│   │   ├── predict_V2_M.py                  # EfficientNet 추론 + Gemini 폴백 + auto_labeled 저장
│   │   └── test_V2_M.py                     # 전체/클래스별 정확도 평가
│   ├── receipt_ocr/
│   │   └── receipt_ocr.py                   # Document AI 1차 + Gemini 2차 정제
│   └── object_detection/
│       └── fridge_detection.py              # Gemini Vision 냉장고 감지
│
├── training/
│   └── train_EfficientNet_V2_M.py           # Progressive Unfreezing 학습 스크립트
│
├── scripts/                                 # 데이터 파이프라인
│   ├── data_crawl.py                        # DuckDuckGo 크롤링 (ddgs)
│   ├── data_clean.py                        # MD5 중복 제거 (dataset 전체)
│   ├── data_split.py                        # train → val 분리 (멱등성 보장)
│   ├── data_diet.py                         # 클래스별 이미지 수 상한
│   └── data_len.py                          # 클래스별 통계 출력
│
├── docs/
│   ├── git-convention.md                    # 커밋·브랜치 컨벤션
│   ├── training_log_5_3.csv                 # ver2 학습 로그
│   ├── training_log_5_13.csv                # ver3 학습 로그
│   ├── training_log_5_27.csv                # ver4 학습 로그 (최신)
│   └── training_log_class_acc_5_27.csv      # ver4 클래스별 정확도
│
├── receipt_model/                           # GCP 서비스 계정 키 (git 제외)
│   └── receipt-app-*.json
│
├── picture_model/                           # 로컬 단독 실행 테스트 이미지 (git 제외)
│   └── predict/
│
├── dataset/                                 # 학습 데이터 (git 제외)
│   ├── train/, val/, test/
│   ├── test_real_image/                     # 영수증 OCR 단독 테스트
│   ├── crawldata/                           # 크롤링 원본
│   └── auto_labeled/                        # Gemini 자동 라벨링 결과
│
└── server.log                               # uvicorn 런타임 로그 (git 제외)
```
</details>

---

## 기술 스택

| 분류 | 라이브러리 / 서비스 | 버전 |
|---|---|---|
| Python | | 3.10.19 |
| 웹 프레임워크 | FastAPI | 0.136.1 |
| ASGI 서버 | uvicorn | 0.46.0 |
| 딥러닝 | PyTorch | 2.11.0 |
| 이미지 처리 | torchvision · Pillow | 0.26.0 · 12.2.0 |
| OCR | Google Cloud Document AI | 3.14.0 |
| LLM / Vision | Google Gemini API (google-genai) | 1.75.0 |
| 환경 변수 | python-dotenv | 1.2.2 |
| 이미지 크롤러 | ddgs | ≥ 2.0.0 |

전체 의존성 목록은 [`requirements.txt`](requirements.txt) 참고.

---

## 컨벤션

- **커밋·브랜치**: [`docs/git-convention.md`](docs/git-convention.md)
- **AI 코딩 어시스턴트 작업 규칙**: [`CLAUDE.md`](CLAUDE.md)

---

## 라이선스

내부 프로젝트 — 외부 배포 금지.
