# fresh-kitchen-ai-server

fresh-kitchen 앱의 AI 백엔드 서버입니다. FastAPI로 구동되며 세 가지 독립적인 ML 파이프라인을 HTTP API로 제공합니다.

---

## 목차

1. [프로젝트 구조](#프로젝트-구조)
2. [기술 스택](#기술-스택)
3. [환경 설정](#환경-설정)
4. [서버 실행](#서버-실행)
5. [API 명세](#api-명세)
   - [공통 사항](#공통-사항)
   - [음식 분류](#1-음식-분류)
   - [영수증 OCR](#2-영수증-ocr)
   - [냉장고 식재료 감지](#3-냉장고-식재료-감지)
6. [파이프라인 상세](#파이프라인-상세)
   - [음식 분류 파이프라인](#음식-분류-파이프라인)
   - [영수증 OCR 파이프라인](#영수증-ocr-파이프라인)
   - [냉장고 감지 파이프라인](#냉장고-감지-파이프라인)
7. [모델 학습](#모델-학습)
8. [학습 결과](#학습-결과)
9. [스크립트 실행](#스크립트-실행)
10. [보안 및 주의사항](#보안-및-주의사항)

---

## 프로젝트 구조

```
fresh-kitchen-ai-server/
├── main.py                              # FastAPI 앱 진입점, 3개 엔드포인트 정의
├── requirements.txt
├── .env.example
├── best_food_model_v2_m_ver3.pth        # 학습된 모델 가중치 (git 제외)
│
├── models/
│   ├── food_classifier/
│   │   ├── predict_V2_M.py             # EfficientNet 추론 + Gemini 폴백 + 자동 저장
│   │   └── test_V2_M.py                # 클래스별 정확도 평가
│   ├── receipt_ocr/
│   │   └── receipt_ocr.py             # Document AI 1차 추출 + Gemini 2차 정제
│   └── object_detection/
│       └── fridge_detection.py        # Gemini Vision 냉장고 식재료 감지
│
├── training/
│   └── train_EfficientNet_V2_M.py     # EfficientNet V2-M 학습 스크립트
│
├── docs/
│   ├── training_log_5_3.csv           # ver2 학습 로그
│   └── training_log_5_13.csv          # ver3 학습 로그
│
├── picture_model/
│   └── predict/                       # 추론 테스트용 샘플 이미지
│
└── dataset/                           # 학습 데이터 (git 제외)
    ├── train/                         # 클래스별 서브폴더
    ├── val/
    ├── test/
    ├── crawldata/                     # Bing 크롤링 원본
    └── auto_labeled/                  # Gemini 자동 레이블링 결과
```

---

## 기술 스택

| 분류 | 라이브러리 / 서비스 | 버전 |
|---|---|---|
| 웹 프레임워크 | FastAPI | 0.136.1 |
| ASGI 서버 | uvicorn | 0.46.0 |
| 딥러닝 | PyTorch | 2.11.0 |
| 이미지 처리 | torchvision / Pillow | 0.26.0 / 12.2.0 |
| OCR | Google Cloud Document AI | 3.14.0 |
| LLM/Vision | Google Gemini API (google-genai) | 1.75.0 |
| 환경 변수 | python-dotenv | 1.2.2 |
| Python | | 3.10.19 |

---

## 환경 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일에 아래 값을 채워 넣습니다.

```env
# Google Cloud Document AI (영수증 OCR용)
GOOGLE_APPLICATION_CREDENTIALS=./receipt_model/your-service-account-key.json
PROJECT_ID=your-project-id
PROCESSOR_ID=your-processor-id
LOCATION=us

# Google Gemini API (음식 분류 폴백 + 영수증 2차 정제 + 냉장고 감지 공용)
GEMINI_API_KEY=your-gemini-api-key

# API 인증 토큰 (Bearer 방식, 백엔드 팀에 전달)
AI_SECRET_TOKEN=your-secret-token
```

`AI_SECRET_TOKEN` 생성 방법:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

`GOOGLE_APPLICATION_CREDENTIALS`는 상대 경로를 지정해도 코드가 프로젝트 루트 기준으로 절대 경로로 변환합니다.

### 3. 모델 가중치 배치

팀 드라이브에서 `best_food_model_v2_m_ver3.pth`를 받아 프로젝트 루트에 놓습니다. 이 파일은 `.gitignore`로 git에서 제외되어 있습니다.

---

## 서버 실행

```bash
# 개발 모드 (파일 변경 시 자동 재시작)
python3 -m uvicorn main:app --reload --port 8000

# 외부 노출 (ngrok 사용 시)
ngrok http 8000
```

서버가 시작되면 `models/food_classifier/predict_V2_M.py`의 `load_food_model()`, `models/receipt_ocr/receipt_ocr.py`, `models/object_detection/fridge_detection.py` 모듈을 순서대로 `importlib`로 동적 로드합니다. 음식 분류 모델(`best_food_model_v2_m_ver3.pth`)은 이 시점에 메모리에 한 번만 올라갑니다.

- API 문서: http://127.0.0.1:8000/docs
- 로그 파일: `server.log` (UTF-8, 서버와 동일 디렉토리에 생성)

---

## API 명세

### 공통 사항

**인증**

모든 엔드포인트는 HTTP Bearer 토큰 인증을 요구합니다.

```
Authorization: Bearer <AI_SECRET_TOKEN>
```

토큰이 없거나 일치하지 않으면 `401 Unauthorized`를 반환합니다.

**파일 제한**

| 항목 | 제한 |
|---|---|
| 최대 파일 크기 | 10 MB |
| 허용 Content-Type | `image/jpeg`, `image/png`, `image/webp` |

- 파일 크기 초과 시 `413` 반환
- 허용되지 않는 Content-Type 시 `415` 반환

업로드된 파일은 처리 중 OS 임시 디렉토리에 저장되며, 응답 반환 직후 즉시 삭제됩니다.

---

### 1. 음식 분류

```
POST /internal/v1/food-classification
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: <image>
```

식재료 또는 음식 이미지를 받아 클래스를 분류합니다.

**응답 (EfficientNet이 직접 분류한 경우)**

```json
{
  "bestMatch": "Kimchi",
  "confidence": 98.42,
  "top3": [
    { "name": "Kimchi",    "confidence": 98.42 },
    { "name": "Gochujang", "confidence": 1.13 },
    { "name": "Doenjang",  "confidence": 0.31 }
  ],
  "source": "efficientnet"
}
```

**응답 (Gemini 폴백 사용 시)**

```json
{
  "bestMatch": "Tofu",
  "confidence": 54.31,
  "top3": [],
  "source": "gemini",
  "gemini_reason": "흰색 사각형 형태의 두부가 촬영된 이미지",
  "auto_saved": "/path/to/dataset/auto_labeled/Tofu/auto_20260514_153022_123456.jpg"
}
```

**응답 (Gemini 폴백도 실패한 경우)**

```json
{
  "bestMatch": "Beef",
  "confidence": 42.10,
  "top3": [
    { "name": "Beef", "confidence": 42.10 },
    { "name": "Pork", "confidence": 31.50 },
    { "name": "Chicken", "confidence": 18.20 }
  ],
  "source": "efficientnet_fallback"
}
```

`source` 필드 값:

| 값 | 의미 |
|---|---|
| `efficientnet` | EfficientNet이 75% 이상 확신도로 분류 |
| `gemini` | EfficientNet 확신도 < 75%여서 Gemini 2.5 Flash가 분류 |
| `efficientnet_fallback` | EfficientNet 확신도 < 75%이지만 Gemini 호출도 실패하여 EfficientNet 결과 그대로 반환 |

**오류 응답**

| 상태 코드 | 설명 |
|---|---|
| 400 | 이미지 읽기 실패 등 예측 불가 |
| 401 | 인증 토큰 불일치 |
| 413 | 파일 크기 10MB 초과 |
| 415 | 허용되지 않는 파일 형식 |
| 500 | 서버 내부 오류 |

---

### 2. 영수증 OCR

```
POST /internal/v1/receipt-ocr
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: <image>
```

영수증 이미지에서 구매일과 식재료 목록을 추출합니다.

**응답**

```json
{
  "purchasedAt": "2026-05-13",
  "ingredients": ["두부", "계란", "만두", "고등어"]
}
```

- `purchasedAt`: `YYYY-MM-DD` 형식. 날짜를 인식하지 못한 경우 `null`
- `ingredients`: 브랜드명이 제거된 순수 식재료명 배열. 비식재료(카드명, 포인트, 할인, 합계, 부가세, 봉투 등)는 제외

Document AI에서 추출된 텍스트가 없으면 Gemini 호출 없이 `{"purchasedAt": null, "ingredients": []}` 반환합니다.

**오류 응답**

| 상태 코드 | 설명 |
|---|---|
| 401 | 인증 토큰 불일치 |
| 413 | 파일 크기 10MB 초과 |
| 415 | 허용되지 않는 파일 형식 |
| 500 | Document AI 또는 Gemini 호출 실패 |

---

### 3. 냉장고 식재료 감지

```
POST /internal/v1/fridge-detection
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: <image>
```

냉장고 내부 사진에서 식재료를 감지합니다.

**응답**

```json
{
  "items": ["계란", "우유", "두부", "당근", "된장"]
}
```

- `items`: 한글 식재료명 배열. 브랜드명 제거 및 중복 제거된 상태
- Gemini API 오류 또는 타임아웃 시 빈 배열 `[]` 반환

**오류 응답**

| 상태 코드 | 설명 |
|---|---|
| 401 | 인증 토큰 불일치 |
| 413 | 파일 크기 10MB 초과 |
| 415 | 허용되지 않는 파일 형식 |
| 500 | Gemini 호출 실패 |

---

## 파이프라인 상세

### 음식 분류 파이프라인

**모델 구조** (`models/food_classifier/predict_V2_M.py`)

- 기반 모델: `torchvision.models.efficientnet_v2_m`
- 분류기 헤드: `Dropout(p=0.3)` + `Linear(in_features, num_classes)`
- 클래스 수 및 클래스 목록은 체크포인트(`best_food_model_v2_m_ver3.pth`)의 `class_names` 키에 저장됨

**추론 전처리**

```python
transforms.Resize(512)
transforms.CenterCrop(480)
transforms.ToTensor()
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

학습의 val 전처리와 동일한 파이프라인을 사용합니다.

**신뢰도 기반 2단계 분류 흐름**

```
이미지 입력
    │
    ▼
EfficientNet V2-M 추론 (softmax → top-1 확신도)
    │
    ├─ confidence ≥ 75% ──→ 결과 반환 (source: "efficientnet")
    │
    └─ confidence < 75%
            │
            ▼
        Gemini 2.5 Flash 호출 (타임아웃: 30초)
            │
            ├─ 성공 ──→ 결과 반환 (source: "gemini")
            │           + 이미지를 dataset/auto_labeled/{label}/ 에 자동 저장
            │
            └─ 실패 ──→ EfficientNet 결과 반환 (source: "efficientnet_fallback")
```

**Gemini 폴백 상세**

- 사용 모델: `gemini-2.5-flash`
- 입력: 원본 이미지 바이트 + 전체 클래스 목록 문자열 (한 줄에 클래스 하나)
- 출력 형식 강제: `response_mime_type="application/json"` → `{"label": "클래스명", "reason": "이유"}`
- 클래스 목록에 없는 것처럼 보이는 이미지도 가장 유사한 클래스로 분류하도록 프롬프트 설계
- Gemini API 호출은 `ThreadPoolExecutor(max_workers=1)`로 감싸 30초 타임아웃 적용

**자동 레이블링 (Auto-labeling)**

Gemini 분류 성공 시 입력 이미지를 `dataset/auto_labeled/{label}/auto_{YYYYMMDD_HHMMSS_microsec}.{ext}` 경로에 `shutil.copy2`로 복사합니다. 이 데이터는 추후 재학습 시 활용할 수 있습니다.

**하드웨어 가속 우선순위**

```
Apple Silicon MPS → NVIDIA CUDA → CPU
```

모델은 서버 시작 시 한 번만 로드되어 전역 변수에 보관되며, 이후 요청마다 재로드하지 않습니다.

---

### 영수증 OCR 파이프라인

**모듈**: `models/receipt_ocr/receipt_ocr.py`

**1단계: Google Cloud Document AI (`process_receipt_raw`)**

- `documentai.DocumentProcessorServiceClient`로 영수증 이미지를 전송
- `entity.type_ == "line_item"` 엔티티 내 `line_item/description` 속성을 순회
- 정규식 `[^가-힣\s]`로 한글 텍스트만 남기고, 1자 이하 항목과 중복(`dict.fromkeys`) 제거
- 전체 OCR 원문(`result.document.text`)도 함께 반환 (2단계에서 날짜 추출에 활용)

**2단계: Gemini 2.5 Flash (`filter_with_gemini`)**

- 1단계에서 추출된 한글 품목 목록 + 전체 OCR 원문을 프롬프트에 포함
- Gemini에게 두 가지 작업을 지시:
  1. `purchasedAt`: `YYYY-MM-DD` 형식 날짜 추출 (없으면 `null`)
  2. `ingredients`: 실제 식재료·식품·음료만 추출, 브랜드명 제거, 중복 제거, 한글 출력
- `response_mime_type="application/json"` 강제로 JSON 응답 보장
- 응답의 `purchasedAt`은 정규식 `^\d{4}-\d{2}-\d{2}$`로 형식 검증, 불일치 시 `null` 처리
- 30초 타임아웃 (ThreadPoolExecutor 사용)
- 1단계 결과(품목 목록, OCR 원문 모두 비어 있는 경우)에는 Gemini 호출을 건너뜀

---

### 냉장고 감지 파이프라인

**모듈**: `models/object_detection/fridge_detection.py`

- Gemini 2.5 Flash Vision API 단일 호출
- 입력: 냉장고 사진 바이트 (`mimetypes.guess_type`으로 MIME 타입 자동 감지) + 텍스트 프롬프트
- 프롬프트 규칙:
  - 식재료, 식품, 음료만 포함
  - 브랜드명 제거 (예: "풀무원 두부" → "두부")
  - 한글로만 출력
  - 불확실한 항목 제외
  - 동의어 중복 제거 (예: "계란"과 "달걀" → "계란")
- `response_mime_type="application/json"` 강제로 JSON 배열 반환
- 응답을 파이썬에서 추가 중복 제거 (문자열 `strip` 기준)
- 30초 타임아웃, 실패 시 빈 배열 반환
- GPU 미사용 (Gemini API 호출 방식)

---

## 모델 학습

**스크립트**: `training/train_EfficientNet_V2_M.py`

### 하이퍼파라미터

| 항목 | 값 |
|---|---|
| 기반 모델 | EfficientNet V2-M (ImageNet 사전학습) |
| 배치 크기 | 8 (MPS 안정성 기준) |
| Learning Rate (초기값) | 1e-4 |
| Optimizer | AdamW (weight_decay=5e-3) |
| LR Scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=3) |
| 최대 Epoch | 30 |
| Early Stopping patience | 5 (val_acc 기준) |
| 드롭아웃 | 0.3 |
| 입력 크기 | 480×480 |
| num_workers | 0 (MPS 멀티프로세싱 비활성화) |

### 파인튜닝 전략

1. ImageNet 사전학습 가중치(`EfficientNet_V2_M_Weights.DEFAULT`)로 모델 초기화
2. 전체 파라미터를 freeze (`requires_grad = False`)
3. `model.features`의 마지막 4개 블록(`features[-4:]`)만 unfreeze
4. 분류기 헤드를 `Dropout(0.3) + Linear(in_features, num_classes)`로 교체
5. Optimizer는 `requires_grad=True`인 파라미터만 대상으로 생성

### 학습 데이터 증강 (Train)

스마트폰 촬영 환경(원근 왜곡, 다양한 조명, 흐림)을 모사하기 위한 구성입니다.

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

### 검증 데이터 전처리 (Val / Test)

추론 전처리(`predict_V2_M.py`)와 동일합니다.

```python
Resize(512)
CenterCrop(480)
Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

### 클래스 불균형 처리

`WeightedRandomSampler`를 사용합니다. 각 클래스의 샘플 수에 반비례하는 가중치 `1 / class_sample_count`를 부여하여, 샘플 수가 적은 클래스가 배치에 더 자주 포함되도록 합니다.

### 체크포인트 저장 형식

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names   # 추론 시 클래스 복원에 필수
}, SAVE_PATH)
```

체크포인트에 클래스 목록이 포함되므로, 추론 시 `dataset/` 폴더 없이도 클래스 정보를 복원할 수 있습니다.

### 손상 이미지 자동 제거

학습 시작 전 `remove_corrupted_images()`가 `dataset/` 하위 모든 이미지를 PIL로 열어 로딩을 시도하고, `IOError` 또는 `SyntaxError` 발생 이미지를 자동 삭제합니다.

### 학습 로그

에폭마다 `docs/training_log.csv`에 `epoch, train_loss, train_acc, val_loss, val_acc, lr`을 기록합니다.

---

## 학습 결과

### ver2 (`docs/training_log_5_3.csv`)

- 학습 환경: Windows, NVIDIA GPU (CUDA)
- 클래스 수: 35개
- 데이터셋: Train 7,315장 / Val 1,803장
- 전체 30 Epoch 완료 (Early Stopping 미발동)
- **최고 val_acc: 94.56%** (Epoch 27)

클래스별 val 정확도 (낮은 순 → 높은 순):

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

- 학습 환경: macOS, Apple Silicon MPS
- 22 Epoch에서 Early Stopping (patience=5)
- **최고 val_acc: 94.73%** (Epoch 17)
- LR은 ReduceLROnPlateau에 의해 1e-4 → 5e-5 → 2.5e-5로 자동 감소

---

## 스크립트 실행

### 단독 추론 테스트

각 모듈은 `if __name__ == "__main__":` 블록이 있어 서버 없이 단독으로 실행할 수 있습니다.

```bash
# 음식 분류 (picture_model/predict/beef1.jpeg 기본값)
python models/food_classifier/predict_V2_M.py

# 냉장고 감지 (picture_model/predict/fridge_test.jpeg)
python models/object_detection/fridge_detection.py

# 영수증 OCR (dataset/test_real_image/receipt.png)
python models/receipt_ocr/receipt_ocr.py
```

### 모델 정확도 평가

```bash
# dataset/test/ 기준 전체 및 클래스별 정확도 출력
python models/food_classifier/test_V2_M.py
```

### 모델 학습

```bash
# dataset/train/, dataset/val/ 폴더가 필요합니다
python training/train_EfficientNet_V2_M.py
```

---

## 보안 및 주의사항

### git에 커밋하지 말아야 할 파일

| 파일 | 이유 |
|---|---|
| `.env` | `GEMINI_API_KEY`, `AI_SECRET_TOKEN`, GCP 프로젝트 정보 포함 |
| `receipt-app-*.json` | Google Cloud 서비스 계정 키 |
| `*.pth` / `*.pt` | 200MB 이상 모델 가중치 |
| `dataset/` | 학습 이미지 |

위 항목들은 `.gitignore`에 등록되어 있습니다.

### API 토큰 관리

`AI_SECRET_TOKEN`은 백엔드 팀과 공유하는 인증 수단입니다. 유출 시 즉시 새 토큰을 생성하여 `.env`에 반영하고 서버를 재시작하세요.

### 파일 업로드 검증

- Content-Type 헤더 기준으로 `image/jpeg`, `image/png`, `image/webp`만 허용합니다.
- 파일 크기는 10MB로 제한됩니다.
- 업로드된 파일은 처리 완료 후 `os.unlink`로 즉시 삭제됩니다.
