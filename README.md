# fresh-kitchen-ai-server

> 냉장고 식재료 관리 앱 **fresh-kitchen** 의 AI 백엔드 서버.
> 한 장의 사진으로 **음식 분류 · 영수증 인식 · 냉장고 인벤토리 자동 구축** 을 수행합니다.

[![Python](https://img.shields.io/badge/Python-3.10.19-blue)]() [![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688)]() [![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C)]()

---

## 목차

1. [기능](#기능)
2. [빠른 시작](#빠른-시작)
3. [환경 변수](#환경-변수)
4. [API 레퍼런스](#api-레퍼런스)
5. [모델 학습](#모델-학습)
6. [데이터 파이프라인](#데이터-파이프라인)
7. [프로젝트 구조](#프로젝트-구조)
8. [기술 스택](#기술-스택)
9. [컨벤션 / 라이선스](#컨벤션--라이선스)

---

## 기능

| 엔드포인트 | 입력 | 출력 | 사용 모델 |
|---|---|---|---|
| `POST /internal/v1/food-classification` | 식재료 사진 1장 | 분류명 + Top3 + 카테고리 | EfficientNet V2-M (70 클래스) + Gemini 폴백 |
| `POST /internal/v1/receipt-ocr` | 영수증 사진 1장 | 구매일 + 식재료 목록 + 카테고리 | Google Document AI + Gemini 2.5 Flash |
| `POST /internal/v1/fridge-detection` | 냉장고 내부 사진 1장 | 식재료 목록 + 카테고리 | Gemini 2.5 Flash Vision |

---

## 빠른 시작

### 사전 요구사항

- Python **3.10.19**
- Google Cloud 계정 (Document AI 활성화) · Gemini API 키
- (선택) NVIDIA GPU — Mixed Precision 학습용

### 설치 및 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일 열고 API 키 입력 (아래 [환경 변수] 참고)

# 3. 모델 가중치 배치 (팀 드라이브에서 다운로드)
# best_food_model_v2_m_ver5.pth → 프로젝트 루트

# 4. 서버 실행
python3 -m uvicorn main:app --reload --port 8000
```

| 항목 | 주소 / 경로 |
|---|---|
| Swagger UI | http://127.0.0.1:8000/docs |
| 서버 로그 | `./server.log` |
| 외부 노출 (선택) | `ngrok http 8000` |

### 동작 확인

```bash
TOKEN=$(grep AI_SECRET_TOKEN .env | cut -d= -f2)

curl -X POST http://127.0.0.1:8000/internal/v1/food-classification \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@samples/predict/beef1.jpeg"
```

---

## 환경 변수

`.env.example`을 복사한 뒤 모든 값을 채워주세요. **`.env` 파일은 절대 git에 커밋되지 않습니다.**

```env
# Google Cloud Document AI (영수증 OCR)
GOOGLE_APPLICATION_CREDENTIALS=./credentials/your-service-account-key.json
PROJECT_ID=your-gcp-project-id
PROCESSOR_ID=your-document-ai-processor-id
LOCATION=us

# Google Gemini API (3개 파이프라인 공통)
GEMINI_API_KEY=your-gemini-api-key

# API 인증 토큰 (Bearer 방식)
AI_SECRET_TOKEN=your-secret-token

# Gemini 호출 타임아웃 (초, 기본값 60 — 생략 가능)
GEMINI_TIMEOUT=60
```

| 변수 | 필수 | 설명 |
|---|:---:|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | ✓ | GCP 서비스 계정 키 경로 |
| `PROJECT_ID` / `PROCESSOR_ID` / `LOCATION` | ✓ | Document AI 프로세서 정보 |
| `GEMINI_API_KEY` | ✓ | Gemini API 키 |
| `AI_SECRET_TOKEN` | ✓ | Bearer 토큰 (누락 시 서버 시작 거부) |
| `GEMINI_TIMEOUT` |  | Gemini 호출 타임아웃 (기본 60초) |

**토큰 생성**:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## API 레퍼런스

### 공통

- 모든 엔드포인트는 `Authorization: Bearer <AI_SECRET_TOKEN>` 헤더 필수
- 요청: `multipart/form-data` 의 `file` 필드
- 파일 제한: 최대 **10 MB**, MIME `image/jpeg | image/png | image/webp`
- 카테고리 enum: `VEGETABLE | FRUIT | MEAT | SEAFOOD | DAIRY | GRAIN | SAUCE | DRINK | ETC`

| 상태 코드 | 상황 |
|---|---|
| `401` | 토큰 없음·불일치 |
| `413` | 파일 10MB 초과 |
| `415` | 허용되지 않는 형식 |
| `500` | 서버 내부 오류 (`server.log` 에서 상세 확인) |

### 1️⃣ 음식 분류 — `POST /internal/v1/food-classification`

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

`source` 값:
- `efficientnet` — 분류기가 80% 이상 확신
- `gemini` — 분류기 확신 부족 → Gemini 폴백 성공 (`dataset/auto_labeled/` 자동 저장)
- `efficientnet_fallback` — Gemini도 실패 → 분류기 원래 결과 사용

### 2️⃣ 영수증 OCR — `POST /internal/v1/receipt-ocr`

```json
{
  "purchasedAt": "2026-05-13",
  "ingredients": [
    { "name": "두부",   "category": "ETC"     },
    { "name": "고등어", "category": "SEAFOOD" }
  ]
}
```

- `purchasedAt`: `YYYY-MM-DD` 형식. 인식 실패 시 `null`
- Document AI 1차 추출 → Gemini 2차 정제 (브랜드명 제거 · 비식재료 필터링)

### 3️⃣ 냉장고 식재료 감지 — `POST /internal/v1/fridge-detection`

```json
{
  "items": [
    { "name": "계란", "category": "ETC"       },
    { "name": "우유", "category": "DAIRY"     },
    { "name": "당근", "category": "VEGETABLE" }
  ]
}
```

Gemini 오류·타임아웃 시 `{"items": []}` 반환 (재시도 없음).

---

## 모델 학습

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
| 최대 Epoch | 50 |
| Early Stopping | val_acc 기준 patience=8 |
| Mixed Precision (AMP) | CUDA에서만 활성화 |

### 학습 데이터 증강

스마트폰 촬영 환경(흐림·역광·기울어짐)을 시뮬레이션하는 증강이 적용되어 있습니다.

| 증강 | 시뮬레이션 대상 |
|---|---|
| `RandomPerspective` | 비스듬한 촬영 각도 |
| `GaussianBlur` | 초점 흐림 |
| `ColorJitter` | 조명·채도 변화 |
| `RandomRotation(30°)` | 회전된 사진 |

### 새 클래스 추가 시 주의사항

1. [데이터 파이프라인](#데이터-파이프라인) 5단계 실행
2. **`models/food_classifier/predict_V2_M.py` 의 `CLASS_CATEGORY` dict에 새 클래스 → 카테고리 매핑 추가** (누락 시 자동으로 `ETC` 처리되며 경고 로그 출력)
3. 학습 스크립트 재실행 후 `.pth` 교체

### 평가

```bash
python models/food_classifier/test_V2_M.py
```

`dataset/test/` 기준 전체 정확도와 클래스별 정확도 출력. 학습 로그는 `docs/training_log_*.csv` 에 저장됩니다.

---

## 데이터 파이프라인

새 클래스 데이터를 추가할 때 아래 순서로 실행:

```bash
python scripts/data_crawl.py   # 1. DuckDuckGo 이미지 크롤링 (ddgs 패키지)
python scripts/data_clean.py   # 2. MD5 해시로 중복 이미지 제거
python scripts/data_split.py   # 3. train → val 분리 (멱등성 보장)
python scripts/data_diet.py    # 4. 클래스별 이미지 수 상한 (train ≤ 400, val ≤ 80)
python scripts/data_len.py     # 5. 최종 통계 출력
```

### 데이터셋 구조

```
dataset/
├── train/<class>/*.jpg     # 학습 데이터
├── val/<class>/*.jpg       # 검증 데이터
├── test/<class>/*.jpg      # 테스트 데이터
├── test_real_image/        # 3개 파이프라인 실제 테스트용 이미지
├── crawldata/              # 크롤링 원본
└── auto_labeled/<label>/   # Gemini 자동 레이블링 결과
```

---

## 프로젝트 구조

```
fresh-kitchen-ai-server/
├── main.py                                  # FastAPI 진입점, 3개 엔드포인트
├── requirements.txt                         # 의존성
├── .env.example                             # 환경 변수 템플릿
├── README.md
├── CLAUDE.md                                # AI 코딩 어시스턴트 작업 규칙
├── DEVLOG.md                                # 개발 의사결정·문제 해결·모델 진화 기록
├── best_food_model_v2_m_ver5.pth            # 운영 모델 (git 제외, ~200MB)
│
├── models/
│   ├── category.py                          # 카테고리 enum + normalize_category()
│   ├── food_classifier/
│   │   ├── predict_V2_M.py                  # EfficientNet 추론 + Gemini 폴백
│   │   └── test_V2_M.py                     # 정확도 평가
│   ├── receipt_ocr/
│   │   └── receipt_ocr.py                   # Document AI + Gemini OCR
│   └── object_detection/
│       └── fridge_detection.py              # Gemini Vision 냉장고 감지
│
├── training/
│   └── train_EfficientNet_V2_M.py           # Progressive Unfreezing 학습 스크립트
│
├── scripts/                                 # 데이터 파이프라인
│   ├── data_crawl.py
│   ├── data_clean.py
│   ├── data_split.py
│   ├── data_diet.py
│   └── data_len.py
│
├── docs/
│   ├── git-convention.md                    # 커밋·브랜치 컨벤션
│   └── training_log_*.csv                   # 학습 로그
│
├── credentials/                            # GCP 서비스 계정 키 (git 제외)
├── samples/                                # 로컬 단독 실행 테스트 이미지 (git 제외)
├── dataset/                                 # 학습 데이터 (git 제외)
└── server.log                               # uvicorn 런타임 로그 (git 제외)
```

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

전체 의존성은 [`requirements.txt`](requirements.txt) 참고.

---

## 컨벤션 / 라이선스

- **커밋·브랜치 컨벤션**: [`docs/git-convention.md`](docs/git-convention.md)
- **AI 코딩 어시스턴트 작업 규칙**: [`CLAUDE.md`](CLAUDE.md)
- **라이선스**: 내부 프로젝트 — 외부 배포 금지

### 보안 — 절대 커밋 금지

`.env` · `receipt-app-*.json` · `credentials/` · `*.pth` / `*.pt` · `dataset/` · `samples/` — 모두 `.gitignore` 등록됨.
