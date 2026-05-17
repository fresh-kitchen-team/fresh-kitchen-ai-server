# fresh-kitchen-ai-server

fresh-kitchen 앱의 AI 백엔드 서버입니다.  
FastAPI로 구동되며 세 가지 독립적인 ML 파이프라인을 HTTP API로 제공합니다.

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [프로젝트 구조](#프로젝트-구조)
3. [기술 스택](#기술-스택)
4. [환경 설정](#환경-설정)
5. [서버 실행](#서버-실행)
6. [API 명세](#api-명세)
   - [공통 사항](#공통-사항)
   - [음식 분류](#1-음식-분류)
   - [영수증 OCR](#2-영수증-ocr)
   - [냉장고 식재료 감지](#3-냉장고-식재료-감지)
7. [파이프라인 상세](#파이프라인-상세)
   - [음식 분류 파이프라인](#음식-분류-파이프라인)
   - [영수증 OCR 파이프라인](#영수증-ocr-파이프라인)
   - [냉장고 감지 파이프라인](#냉장고-감지-파이프라인)
8. [모델 학습](#모델-학습)
9. [학습 결과](#학습-결과)
10. [데이터 파이프라인 스크립트](#데이터-파이프라인-스크립트)
11. [보안 및 주의사항](#보안-및-주의사항)

---

## 프로젝트 개요

| 파이프라인 | 기술 | 역할 |
|---|---|---|
| 음식 분류 | EfficientNet V2-M + Gemini 2.5 Flash (폴백) | 식재료 이미지를 60개 클래스로 분류 |
| 영수증 OCR | Google Document AI + Gemini 2.5 Flash | 영수증에서 구매일·식재료 목록 추출 |
| 냉장고 감지 | Gemini 2.5 Flash Vision | 냉장고 사진에서 식재료 목록 감지 |

**신뢰도 기반 자동 폴백**: 음식 분류 모델의 확신도가 75% 미만이면 Gemini Vision이 대신 판단하고, 분류 결과를 `dataset/auto_labeled/`에 자동 저장하여 이후 재학습에 활용합니다.

---

## 프로젝트 구조

```
fresh-kitchen-ai-server/
├── main.py                               # FastAPI 앱 진입점, 3개 엔드포인트
├── requirements.txt
├── .env.example
├── best_food_model_v2_m_ver3.pth         # 학습된 모델 가중치 (git 제외)
│
├── models/
│   ├── category.py                      # 카테고리 상수 및 정규화 함수 공통 모듈
│   ├── food_classifier/
│   │   ├── predict_V2_M.py              # EfficientNet 추론 + Gemini 폴백 + 자동 저장
│   │   └── test_V2_M.py                 # 클래스별 정확도 평가
│   ├── receipt_ocr/
│   │   └── receipt_ocr.py              # Document AI 1차 추출 + Gemini 2차 정제
│   └── object_detection/
│       └── fridge_detection.py         # Gemini Vision 냉장고 식재료 감지
│
├── training/
│   └── train_EfficientNet_V2_M.py      # EfficientNet V2-M 학습 스크립트
│
├── scripts/
│   ├── data_crawl.py                   # Bing/Google 이미지 크롤링
│   ├── data_clean.py                   # MD5 해시 기반 중복 이미지 제거
│   ├── data_split.py                   # train/val 분리
│   ├── data_diet.py                    # 클래스별 이미지 수 상한 조정
│   ├── data_blurred.py                 # 블러 처리 val 데이터셋 생성
│   └── data_len.py                     # 클래스별 이미지 수 통계 출력
│
├── docs/
│   ├── training_log_5_3.csv            # ver2 에폭별 학습 로그
│   ├── training_log_5_13.csv           # ver3 에폭별 학습 로그
│   └── git-convention.md               # 브랜치·커밋 컨벤션
│
├── receipt_model/                      # Google Cloud 서비스 계정 키 (git 제외)
│
└── dataset/                            # 학습 데이터 (git 제외)
    ├── train/                          # 클래스별 서브폴더
    ├── val/
    ├── test/
    ├── crawldata/                      # 크롤링 원본 이미지
    └── auto_labeled/                   # Gemini 자동 레이블링 결과
```

---

## 기술 스택

| 분류 | 라이브러리 / 서비스 | 버전 |
|---|---|---|
| Python | | 3.10.19 |
| 웹 프레임워크 | FastAPI | 0.136.1 |
| ASGI 서버 | uvicorn | 0.46.0 |
| 딥러닝 프레임워크 | PyTorch | 2.11.0 |
| 이미지 처리 | torchvision / Pillow | 0.26.0 / 12.2.0 |
| OCR | Google Cloud Document AI | 3.14.0 |
| LLM / Vision | Google Gemini API (google-genai) | 1.75.0 |
| 환경 변수 | python-dotenv | 1.2.2 |

---

## 환경 설정

### 1. Python 버전 확인

```bash
python3 --version  # 3.10.19 권장
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일에 아래 값을 모두 채웁니다.

```env
# Google Cloud Document AI (영수증 OCR용)
GOOGLE_APPLICATION_CREDENTIALS=./receipt_model/your-service-account-key.json
PROJECT_ID=your-gcp-project-id
PROCESSOR_ID=your-document-ai-processor-id
LOCATION=us

# Google Gemini API (음식 분류 폴백 · 영수증 2차 정제 · 냉장고 감지 공용)
GEMINI_API_KEY=your-gemini-api-key

# API 인증 토큰 (Bearer 방식, 백엔드 팀에 전달)
AI_SECRET_TOKEN=your-secret-token

# Gemini API 타임아웃 (초, 기본값 30 — 생략 가능)
GEMINI_TIMEOUT=30
```

> **`AI_SECRET_TOKEN` 생성 방법**
> ```bash
> python3 -c "import secrets; print(secrets.token_hex(32))"
> ```

> **참고**: `GOOGLE_APPLICATION_CREDENTIALS`에 상대 경로를 지정해도 코드가 프로젝트 루트 기준 절대 경로로 자동 변환합니다.

> **주의**: `AI_SECRET_TOKEN`이 `.env`에 없으면 서버가 시작을 거부합니다(`RuntimeError`).

### 4. 모델 가중치 배치

팀 드라이브에서 `best_food_model_v2_m_ver3.pth`를 받아 프로젝트 루트에 놓습니다.  
이 파일은 `.gitignore`에 등록되어 git에서 제외됩니다.

---

## 서버 실행

```bash
# 개발 모드 (파일 변경 시 자동 재시작)
python3 -m uvicorn main:app --reload --port 8000

# 외부 노출 (ngrok 사용 시)
ngrok http 8000
```

서버가 시작되면 `lifespan` 훅에서 `load_food_model()`을 호출해 EfficientNet 모델을 메모리에 로드합니다. Gemini와 Document AI 클라이언트는 모듈 임포트 시점에 싱글톤으로 초기화되어 요청마다 재사용됩니다.

| 항목 | 주소 |
|---|---|
| Swagger UI (API 문서) | http://127.0.0.1:8000/docs |
| 서버 로그 파일 | 프로젝트 루트 `server.log` (UTF-8) |

**하드웨어 가속 우선순위**: Apple Silicon MPS → NVIDIA CUDA → CPU  
(음식 분류 모델 및 학습 스크립트에 적용, 냉장고 감지는 Gemini API 호출이므로 미적용)

---

## API 명세

### 공통 사항

#### 인증

모든 엔드포인트는 HTTP Bearer 토큰 인증을 요구합니다.

```
Authorization: Bearer <AI_SECRET_TOKEN>
```

| 상황 | 상태 코드 |
|---|---|
| 토큰 없음 또는 불일치 | `401 Unauthorized` |

#### 파일 제한

| 항목 | 제한 |
|---|---|
| 최대 파일 크기 | 10 MB |
| 허용 MIME 타입 | `image/jpeg`, `image/png`, `image/webp` |

- 파일 크기 초과 → `413`
- 허용되지 않는 형식 → `415`

Content-Type 헤더 외에 실제 파일 바이트(magic bytes)로도 형식을 검증합니다.  
업로드된 파일은 OS 임시 디렉토리에 저장되고, 응답 반환 직후 즉시 삭제됩니다.

#### 카테고리 값

세 API 모두 `category` 필드에 아래 값 중 하나를 반환합니다.

| 값 | 의미 |
|---|---|
| `VEGETABLE` | 채소 |
| `FRUIT` | 과일 |
| `MEAT` | 육류 |
| `SEAFOOD` | 해산물 |
| `DAIRY` | 유제품 |
| `GRAIN` | 곡물 / 면류 |
| `SAUCE` | 소스 / 양념 |
| `DRINK` | 음료 |
| `ETC` | 기타 |

카테고리 상수와 정규화 로직은 `models/category.py`에서 관리합니다. Gemini가 목록에 없는 값을 반환하면 자동으로 `ETC`로 변환됩니다.

---

### 1. 음식 분류

```
POST /internal/v1/food-classification
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `file` | File (multipart) | 분류할 식재료 또는 음식 이미지 |

#### 응답 — EfficientNet이 직접 분류 (`confidence ≥ 75%`)

```json
{
  "bestMatch": "Kimchi",
  "category": "VEGETABLE",
  "confidence": 98.42,
  "top3": [
    { "name": "Kimchi",    "confidence": 98.42 },
    { "name": "Gochujang", "confidence":  1.13 },
    { "name": "Doenjang",  "confidence":  0.31 }
  ],
  "source": "efficientnet"
}
```

#### 응답 — Gemini 폴백 사용 (`confidence < 75%`, Gemini 성공)

```json
{
  "bestMatch": "Tofu",
  "category": "ETC",
  "confidence": 54.31,
  "top3": [],
  "source": "gemini",
  "auto_saved": "/abs/path/to/dataset/auto_labeled/Tofu/auto_20260514_153022_123456.jpg"
}
```

#### 응답 — Gemini 폴백도 실패 (`confidence < 75%`, Gemini 실패)

```json
{
  "bestMatch": "Beef",
  "category": "MEAT",
  "confidence": 42.10,
  "top3": [
    { "name": "Beef",    "confidence": 42.10 },
    { "name": "Pork",    "confidence": 31.50 },
    { "name": "Chicken", "confidence": 18.20 }
  ],
  "source": "efficientnet_fallback"
}
```

#### `source` 필드 값

| 값 | 의미 |
|---|---|
| `efficientnet` | EfficientNet이 75% 이상 확신도로 분류 |
| `gemini` | 확신도 < 75%여서 Gemini 2.5 Flash가 분류. `auto_saved` 필드에 자동 저장 경로 포함 |
| `efficientnet_fallback` | 확신도 < 75%이지만 Gemini 호출도 실패. EfficientNet 결과를 그대로 반환 |

#### 오류 응답

| 상태 코드 | 설명 |
|---|---|
| `400` | 이미지 읽기 실패 등 예측 불가 상태 |
| `401` | 인증 토큰 불일치 |
| `413` | 파일 크기 10 MB 초과 |
| `415` | 허용되지 않는 파일 형식 |
| `500` | 서버 내부 오류 |

---

### 2. 영수증 OCR

```
POST /internal/v1/receipt-ocr
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `file` | File (multipart) | 영수증 이미지 |

영수증 이미지에서 구매일과 식재료 목록을 추출합니다.

#### 응답

```json
{
  "purchasedAt": "2026-05-13",
  "ingredients": [
    { "name": "두부",   "category": "ETC"       },
    { "name": "계란",   "category": "ETC"       },
    { "name": "만두",   "category": "GRAIN"     },
    { "name": "고등어", "category": "SEAFOOD"   }
  ]
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `purchasedAt` | `string \| null` | `YYYY-MM-DD` 형식. 날짜 인식 불가 시 `null` |
| `ingredients` | `array` | 식재료 객체 배열. 브랜드명 제거·중복 제거된 상태 |
| `ingredients[].name` | `string` | 한글 식재료명 |
| `ingredients[].category` | `string` | 카테고리 (위 [카테고리 값](#카테고리-값) 참고) |

Document AI에서 텍스트를 전혀 추출하지 못하면 Gemini 호출 없이 `{"purchasedAt": null, "ingredients": []}` 반환합니다.

비식재료(카드명, 포인트, 환불, 할인, 합계, 부가세, 봉투 등)는 Gemini 2차 정제 단계에서 제외됩니다.

#### 오류 응답

| 상태 코드 | 설명 |
|---|---|
| `401` | 인증 토큰 불일치 |
| `413` | 파일 크기 10 MB 초과 |
| `415` | 허용되지 않는 파일 형식 |
| `500` | Document AI 또는 Gemini 호출 실패 |

---

### 3. 냉장고 식재료 감지

```
POST /internal/v1/fridge-detection
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `file` | File (multipart) | 냉장고 내부 사진 |

냉장고 내부 사진에서 식재료를 감지합니다.

#### 응답

```json
{
  "items": [
    { "name": "계란",   "category": "ETC"      },
    { "name": "우유",   "category": "DAIRY"    },
    { "name": "두부",   "category": "ETC"      },
    { "name": "당근",   "category": "VEGETABLE"},
    { "name": "된장",   "category": "SAUCE"    }
  ]
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `items` | `array` | 감지된 식재료 객체 배열 |
| `items[].name` | `string` | 한글 식재료명 (브랜드명 제거·중복 제거) |
| `items[].category` | `string` | 카테고리 (위 [카테고리 값](#카테고리-값) 참고) |

Gemini API 오류 또는 타임아웃 시 빈 배열 `{ "items": [] }` 반환합니다.

#### 오류 응답

| 상태 코드 | 설명 |
|---|---|
| `401` | 인증 토큰 불일치 |
| `413` | 파일 크기 10 MB 초과 |
| `415` | 허용되지 않는 파일 형식 |
| `500` | Gemini 호출 실패 |

---

## 파이프라인 상세

### 음식 분류 파이프라인

**모듈**: `models/food_classifier/predict_V2_M.py`

#### 모델 구조

- 기반 모델: `torchvision.models.efficientnet_v2_m` (ImageNet 사전학습 가중치)
- 분류기 헤드: `Dropout(p=0.3)` + `Linear(in_features, num_classes)`
- 클래스 수·클래스 목록은 체크포인트의 `class_names` 키에 저장되므로, 추론 시 `dataset/` 폴더 없이 복원 가능

#### 추론 전처리

학습의 val 전처리와 동일합니다.

```python
Resize(512) → CenterCrop(480) → ToTensor() → Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

#### 신뢰도 기반 2단계 분류 흐름

```
이미지 입력
    │
    ▼
EfficientNet V2-M 추론 (softmax × 100 → top-1 확신도)
    │
    ├─ confidence ≥ 75% ───────────────────────────────────────► 반환 (source: "efficientnet")
    │
    └─ confidence < 75%
            │
            ▼
        Gemini 2.5 Flash 호출 (타임아웃: GEMINI_TIMEOUT초)
            │
            ├─ 성공 ──► 반환 (source: "gemini")
            │           + 이미지 → dataset/auto_labeled/{label}/ 자동 저장
            │
            └─ 실패 ──► EfficientNet 결과 반환 (source: "efficientnet_fallback")
```

#### Gemini 폴백 상세

- 입력: 원본 이미지 바이트 + 전 클래스 목록 + 카테고리 목록
- 출력: `response_mime_type="application/json"` 강제 → `{"label": "클래스명", "category": "카테고리"}`
- 클래스 목록에 없는 것처럼 보이는 이미지도 가장 유사한 클래스로 분류하도록 프롬프트 설계
- `ThreadPoolExecutor(max_workers=1)` + `future.result(timeout=...)` 으로 타임아웃 제어

#### 자동 레이블링 (Auto-labeling)

Gemini 분류 성공 시 입력 이미지를 아래 경로에 `shutil.copy2`로 복사합니다.

```
dataset/auto_labeled/{label}/auto_{YYYYMMDD_HHMMSS_microsec}.{ext}
```

이 데이터는 추후 재학습 시 학습 데이터로 편입할 수 있습니다.

---

### 영수증 OCR 파이프라인

**모듈**: `models/receipt_ocr/receipt_ocr.py`

#### 1단계 — Google Cloud Document AI (`process_receipt_raw`)

1. `documentai.DocumentProcessorServiceClient`로 영수증 이미지 전송
2. `entity.type_ == "line_item"` 내 `line_item/description` 속성을 순회
3. 정규식 `[^가-힣\s]`로 한글 텍스트만 추출, 1자 이하·중복 제거 (`dict.fromkeys`)
4. 전체 OCR 원문(`result.document.text`)도 함께 반환 → 2단계 날짜 추출에 활용

#### 2단계 — Gemini 2.5 Flash (`filter_with_gemini`)

Gemini에게 두 가지 작업을 지시합니다.

- `purchasedAt`: `YYYY-MM-DD` 형식 날짜 추출 (없으면 `null`)
- `ingredients`: 실제 식재료·식품·음료만 추출, 브랜드명 제거, 중복 제거, `category` 분류, 한글 출력

`response_mime_type="application/json"` 강제로 JSON 응답을 보장하며, `purchasedAt`은 정규식 `^\d{4}-\d{2}-\d{2}$`로 재검증합니다.

1단계의 품목 목록과 OCR 원문이 모두 비어 있으면 Gemini 호출을 건너뜁니다.

---

### 냉장고 감지 파이프라인

**모듈**: `models/object_detection/fridge_detection.py`

Gemini 2.5 Flash Vision API 단일 호출로 처리합니다.

- 입력: 냉장고 사진 바이트 + 텍스트 프롬프트 (`mimetypes.guess_type`으로 MIME 타입 자동 감지)
- 프롬프트 규칙:
  - 식재료·식품·음료만 포함
  - 브랜드명 제거 (예: "풀무원 두부" → "두부")
  - 한글로만 출력
  - 불확실한 항목 제외
  - 동의어 중복 제거 (예: "계란"과 "달걀" → "계란")
  - `category` 분류 포함
- `response_mime_type="application/json"` 강제 → JSON 배열 반환
- Python 레벨에서 `name.strip()` 기준으로 추가 중복 제거
- 타임아웃 또는 실패 시 빈 배열 반환

---

## 모델 학습

**스크립트**: `training/train_EfficientNet_V2_M.py`

```bash
# dataset/train/, dataset/val/ 폴더가 준비되어 있어야 합니다
python training/train_EfficientNet_V2_M.py
```

### 하이퍼파라미터

| 항목 | 값 |
|---|---|
| 기반 모델 | EfficientNet V2-M (ImageNet 사전학습) |
| 입력 크기 | 480 × 480 |
| 배치 크기 | 8 (MPS 안정성 기준) |
| Learning Rate (초기값) | 1e-4 |
| Optimizer | AdamW (weight_decay=1e-3) |
| Label Smoothing | 0.1 |
| Gradient Clipping | max_norm=1.0 |
| LR Scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=3) |
| 최대 Epoch | 30 |
| Early Stopping | patience=5 (val_acc 기준) |
| Dropout | 0.3 |
| num_workers | 0 (MPS 멀티프로세싱 비활성화) |

### 파인튜닝 전략 (Progressive Unfreezing)

1. ImageNet 사전학습 가중치(`EfficientNet_V2_M_Weights.DEFAULT`)로 초기화
2. 전체 파라미터 freeze (`requires_grad = False`)
3. `model.features[-4:]` 마지막 4개 블록 + 분류기 헤드 unfreeze (LR = 1e-4)
4. Epoch 5 이후 나머지 전체 레이어 해동, backbone LR = 1e-5 (별도 param_group 추가)

### 학습 데이터 증강 (Train)

스마트폰 촬영 환경(원근 왜곡, 다양한 조명, 흔들림)을 모사하기 위한 구성입니다.

```python
RandomResizedCrop(480, scale=(0.7, 1.0))
RandomHorizontalFlip()
RandomRotation(30)
RandomPerspective(distortion_scale=0.2, p=0.5)
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
RandomAdjustSharpness(sharpness_factor=2, p=0.3)
RandomGrayscale(p=0.05)
Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

### 검증 데이터 전처리 (Val)

추론(`predict_V2_M.py`)과 동일한 전처리를 사용합니다.

```python
Resize(512) → CenterCrop(480) → Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

### 클래스 불균형 처리

`WeightedRandomSampler`를 사용합니다. 각 클래스 샘플 수에 반비례하는 가중치(`1 / class_sample_count`)를 부여해 샘플 수가 적은 클래스가 배치에 더 자주 포함됩니다.

### 손상 이미지 자동 제거

학습 시작 전 `remove_corrupted_images()`가 `dataset/` 하위 모든 이미지를 PIL로 로딩하고, `IOError` 또는 `SyntaxError` 발생 파일을 자동 삭제합니다.

### 체크포인트 저장 형식

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names  # 추론 시 클래스 복원에 필수
}, SAVE_PATH)
```

### 학습 로그 (CSV)

학습 완료 후 `docs/` 아래에 두 개의 CSV가 저장됩니다.

| 파일 | 내용 |
|---|---|
| `training_log.csv` | 에폭별 `train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr_head`, `lr_backbone` |
| `training_log_class_accuracy.csv` | 베스트 모델 기준 클래스별 `correct`, `total`, `accuracy` |

### 단독 모델 평가

```bash
# dataset/test/ 기준 전체 및 클래스별 정확도 출력
python models/food_classifier/test_V2_M.py
```

테스트 폴더의 클래스 순서가 모델의 `class_names`와 다르면 즉시 중단하고 불일치 목록을 출력합니다.

---

## 학습 결과

### ver2 (`docs/training_log_5_3.csv`)

| 항목 | 내용 |
|---|---|
| 학습 환경 | Windows / NVIDIA GPU (CUDA) |
| 클래스 수 | 35개 |
| 데이터셋 | Train 7,315장 / Val 1,803장 |
| 학습 완료 | 전체 30 Epoch (Early Stopping 미발동) |
| **최고 val_acc** | **94.56%** (Epoch 27) |

클래스별 val 정확도:

| 클래스 | 정확도 | 클래스 | 정확도 |
|---|---|---|---|
| Pork | 79.7% | GreenOnion | 96.6% |
| Cheese | 80.4% | ChiliPowder | 96.8% |
| Butter | 85.7% | Lettuce | 97.2% |
| Bacon | 88.0% | SoySauce | 97.5% |
| ChiliPepper | 88.5% | Milk | 97.8% |
| Sausage | 88.5% | Spinach | 98.2% |
| Beef | 89.7% | Doenjang | 98.6% |
| Gochujang | 90.7% | Blueberry | 98.6% |
| Salt | 91.1% | Potato | 98.6% |
| Radish | 91.3% | Vinegar | 98.6% |
| OysterSauce | 92.1% | Mushroom | 98.9% |
| Beansprout | 92.2% | Kimchi | 100.0% |
| Ssamjang | 92.5% | Mustard | 100.0% |
| BellPepper | 94.1% | Shrimp | 100.0% |
| Chicken | 95.2% | Sugar | 100.0% |
| Ketchup | 95.8% | | |
| Mayonnaise | 96.0% | | |
| Egg | 96.2% | | |

### ver3 (`docs/training_log_5_13.csv`)

| 항목 | 내용 |
|---|---|
| 학습 환경 | Windows / NVIDIA GPU (CUDA) |
| 클래스 수 | 60개 |
| 데이터셋 | Train 11,050장 / Val 2,731장 |
| 학습 완료 | 22 Epoch (Early Stopping, patience=5) |
| **최고 val_acc** | **94.73%** (Epoch 17) |
| LR 변화 | 1e-4 → 5e-5 → 2.5e-5 (ReduceLROnPlateau 자동 감소) |

---

## 데이터 파이프라인 스크립트

새 클래스 데이터를 구축할 때 아래 순서로 실행합니다.

```bash
# 1. 이미지 크롤링 (Bing + Google)
python scripts/data_crawl.py

# 2. MD5 해시 기반 중복 이미지 제거
python scripts/data_clean.py

# 3. train/val 분리
python scripts/data_split.py

# 4. 클래스별 이미지 수 상한 조정 (train: 400장, val: 80장)
python scripts/data_diet.py

# 5. 블러 처리 val 데이터셋 생성 (스마트폰 흔들림 모사)
python scripts/data_blurred.py

# 6. 클래스별 이미지 수 통계 출력
python scripts/data_len.py
```

### 단독 추론 테스트

각 모듈은 `if __name__ == "__main__":` 블록이 있어 서버 없이 단독 실행이 가능합니다.

```bash
# 음식 분류 (picture_model/predict/beef1.jpeg)
python models/food_classifier/predict_V2_M.py

# 냉장고 감지 (picture_model/predict/fridge_test.jpeg)
python models/object_detection/fridge_detection.py

# 영수증 OCR (dataset/test_real_image/receipt.png)
python models/receipt_ocr/receipt_ocr.py
```

---

## 보안 및 주의사항

### git에 절대 커밋하지 말아야 할 파일

| 파일 | 이유 |
|---|---|
| `.env` | `GEMINI_API_KEY`, `AI_SECRET_TOKEN`, GCP 프로젝트 정보 포함 |
| `receipt_model/*.json` | Google Cloud 서비스 계정 개인 키 |
| `*.pth` / `*.pt` | 200 MB 이상 모델 가중치 |
| `dataset/` | 학습 이미지 전체 |

위 항목은 모두 `.gitignore`에 등록되어 있습니다.

### API 토큰 관리

`AI_SECRET_TOKEN`은 백엔드 팀과 공유하는 유일한 인증 수단입니다.  
토큰 유출 시 즉시 아래 절차를 따르세요.

1. `python3 -c "import secrets; print(secrets.token_hex(32))"` 로 새 토큰 생성
2. `.env`의 `AI_SECRET_TOKEN` 값 교체
3. 서버 재시작
4. 백엔드 팀에 새 토큰 전달

### 파일 업로드 보안

- **이중 형식 검증**: Content-Type 헤더 + 실제 파일 magic bytes 두 단계로 검증합니다.
- **크기 제한**: 10 MB 초과 파일은 즉시 거부합니다.
- **임시 파일 자동 삭제**: 업로드된 파일은 처리 완료 후 `os.unlink`로 즉시 삭제됩니다.
- **타이밍 공격 방어**: 토큰 비교에 `hmac.compare_digest()`를 사용합니다.
- **내부 오류 비노출**: 500 오류 발생 시 예외 메시지 대신 고정 문자열만 반환합니다. 상세 스택 트레이스는 `server.log`에만 기록됩니다.
