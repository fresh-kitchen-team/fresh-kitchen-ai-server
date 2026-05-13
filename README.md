# fresh-kitchen-ai-server

fresh-kitchen 프로젝트의 AI 서버 — 음식 이미지 분류, 영수증 OCR, 냉장고 물체 감지를 담당합니다.

---

## 전체 시스템 흐름

```
사용자 (앱)
    ↓ 사진 촬영
백엔드 (Spring)
    ↓ 이미지 + Bearer 토큰 전달
AI 서버 (FastAPI) ← main.py
    ↓ 모델 추론
EfficientNet V2-M / Document AI / Gemini Vision
    ↓ JSON 반환
백엔드 (Spring)
    ↓ DTO 변환 후 DB 저장 및 앱 응답
사용자 (앱)
```

---

## 구성

### models/food_classifier — 음식 이미지 분류
- **모델**: EfficientNet V2-M (48개 클래스, Val Acc 94.34%)
- 확신도 75% 미만이면 Gemini Vision으로 자동 위임
- `predict_V2_M.py`: 이미지 예측
- `test_V2_M.py`: 모델 정확도 평가

### models/receipt_ocr — 영수증 OCR
- **기술**: Google Cloud Document AI + Gemini API
- 영수증 이미지 → 텍스트 추출 → 식재료 목록 반환
- `receipt_ocr.py`: 영수증 OCR 처리
- `receipt.png`, `receipt2.jpeg`: 테스트용 영수증 샘플

### models/object_detection — 냉장고 식재료 감지
- **기술**: Gemini Vision API
- 냉장고 사진 → 식재료 목록 JSON 반환
- `fridge_detection.py`: 냉장고 식재료 감지

### training — 모델 학습
- `train_EfficientNet_V2_M.py`: EfficientNet V2-M 학습 (30 Epoch, Early Stopping patience=5)

### scripts — 데이터 전처리 (Git 미포함)

| 파일 | 역할 |
|------|------|
| `data_crawl.py` | Bing에서 음식 이미지 크롤링 |
| `data_clean.py` | 중복/손상 이미지 제거 |
| `data_split.py` | train/val 분할 |
| `data_diet.py` | 클래스당 이미지 수 제한 |
| `data_blurred.py` | 블러 처리 검증 데이터 생성 |
| `data_len.py` | 데이터셋 현황 리포트 |

#### 데이터 파이프라인 실행 순서

```bash
python scripts/data_crawl.py    # 1. 이미지 수집
python scripts/data_clean.py    # 2. 중복/손상 제거
python scripts/data_split.py    # 3. train/val 분할
python scripts/data_diet.py     # 4. 개수 제한
python scripts/data_blurred.py  # 5. 블러 검증 데이터 생성
python scripts/data_len.py      # 6. 현황 리포트
```

### docs — 개발 문서
- `git-convention.md`: 커밋 메시지 / 브랜치 / PR 컨벤션
- `training_log_5_3.csv`: 모델 학습 로그
- `상세개발계획서 AI담당.xlsx`: 개발 계획서

---

## 설치

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
# .env 파일에 API 키와 Bearer 토큰 입력
```

모델 가중치(`best_food_model_v2_m_ver3.pth`)는 Git에 포함되지 않습니다. 팀 드라이브에서 다운로드 후 프로젝트 루트에 저장하세요.

---

## 실행

### AI 서버 (FastAPI)

```bash
python3 -m uvicorn main:app --reload --port 8000
```

서버 시작 후 API 문서 확인:
```
http://127.0.0.1:8000/docs
```

### 외부 접속 (백엔드 팀 연동)

```bash
ngrok http 8000
```
생성된 `https://xxxx.ngrok-free.dev` 주소를 백엔드 팀에게 전달하세요.

> ⚠️ 무료 플랜은 ngrok 재시작 시 주소가 바뀝니다.

### 개별 모델 실행

```bash
# 음식 이미지 분류
python models/food_classifier/predict_V2_M.py

# 영수증 분석
python models/receipt_ocr/receipt_ocr.py

# 냉장고 식재료 감지
python models/object_detection/fridge_detection.py

# 모델 정확도 평가
python models/food_classifier/test_V2_M.py

# 모델 학습
python training/train_EfficientNet_V2_M.py
```

---

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/internal/v1/food-classification` | 식재료 이미지 분류 |
| POST | `/internal/v1/receipt-ocr` | 영수증 OCR → 식재료 추출 |
| POST | `/internal/v1/fridge-detection` | 냉장고 사진 식재료 감지 |

모든 엔드포인트는 `multipart/form-data`로 이미지 파일을 받습니다.

Bearer 토큰 인증 필요: `Authorization: Bearer {AI_SECRET_TOKEN}`

### Bearer 토큰 발급

```bash
# 토큰 생성
python3 -c "import secrets; print(secrets.token_hex(32))"

# .env에 저장
AI_SECRET_TOKEN=생성된_토큰값
```

생성한 토큰을 백엔드 팀에게 전달하면 백엔드에서 모든 요청에 헤더로 포함하여 전송합니다.

### 테스트 방법

**Swagger UI** (브라우저)
1. `http://127.0.0.1:8000/docs` 접속
2. 우측 상단 **Authorize** 클릭 → 토큰 입력
3. 엔드포인트 펼치고 **Try it out** → 이미지 업로드 → **Execute**

**curl**
```bash
curl -X POST http://127.0.0.1:8000/internal/v1/food-classification \
  -H "Authorization: Bearer {AI_SECRET_TOKEN}" \
  -F "file=@picture_model/predict/beef1.jpeg"
```

---

## 디렉토리 구조

```
fresh-kitchen-ai-server/
├── main.py                          # FastAPI 서버 진입점
├── best_food_model_v2_m_ver3.pth    # 학습된 모델 (Git 미포함)
├── models/
│   ├── food_classifier/             # EfficientNet + Gemini 혼합 분류
│   ├── receipt_ocr/                 # Document AI + Gemini OCR
│   └── object_detection/            # Gemini Vision 냉장고 식재료 감지
├── training/                        # 모델 학습 스크립트
├── scripts/                         # 데이터 전처리 유틸리티 (Git 미포함)
├── receipt_model/                   # Google Cloud 인증 파일 (Git 미포함)
├── picture_model/predict/           # 테스트용 이미지 샘플 (Git 미포함)
├── dataset/                         # 학습 데이터 (Git 미포함)
├── docs/                            # 개발 문서 및 학습 로그
├── .env.example
├── requirements.txt
├── CLAUDE.md                        # AI 에이전트 가이드
└── README.md
```

---

## 보안 주의사항

다음 파일은 절대 Git에 올리지 마세요 (`.gitignore`에 자동 제외):

- `receipt-app-*.json` — Google Cloud 서비스 계정 키
- `.env` — API 키 및 Bearer 토큰
- `*.pth`, `*.pt` — 모델 가중치
- `dataset/` — 학습 데이터
- `scripts/` — 데이터 전처리 스크립트
- `picture_model/` — 테스트 이미지
- `receipt_model/` — Google Cloud 인증 파일

---

## 개발 환경

- **Python**: 3.10.19 (conda 환경 권장)
- **PyTorch**: 2.11.0 / TorchVision 0.26.0
- **FastAPI**: 0.136.1 / Uvicorn 0.46.0
- **Google Cloud**: Document AI / Gemini 2.5 Flash
- **Gemini Vision**: 냉장고 식재료 감지

자세한 의존성은 `requirements.txt` 참조.

---

## Git 컨벤션

커밋 / 브랜치 / PR 작성 규칙은 [`docs/git-convention.md`](docs/git-convention.md)를 참조하세요.

요약:
- 커밋: `Type(Scope) : 한국어 설명`
- PR: `Type: Description` (영어, scope 없음)
- 브랜치: `feat/domain-feature`
