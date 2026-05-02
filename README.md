# fresh-kitchen-ai-server

fresh-kitchen 프로젝트의 AI 서버 — 음식 이미지 분류, 영수증 OCR, 냉장고 물체 감지를 담당합니다.

## 전체 시스템 흐름

```
사용자 (앱)
    ↓ 사진 촬영
백엔드 (Spring)
    ↓ 이미지 + Bearer 토큰 전달
AI 서버 (FastAPI) ← main.py
    ↓ 모델 추론
EfficientNet V2-M / Document AI / YOLOv8n
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
- `receipt_ocr.py`

### models/object_detection — 냉장고 물체 감지
- **모델**: YOLOv8n
- 냉장고/음식 물품 탐지
- `yolo_predict.py`

### training — 모델 학습
- `train_EfficientNet_V2_M.py`: EfficientNet V2-M 학습 (30 Epoch, Early Stopping patience=5)

---

## 설치

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
# .env 파일에 자신의 API 키 입력
```

모델 가중치(`best_food_model_v2_m_ver2.pth`)는 Git에 포함되지 않습니다. 팀 드라이브에서 다운로드 후 프로젝트 루트에 저장하세요.

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

# 냉장고 물체 감지
python models/object_detection/yolo_predict.py

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
| POST | `/internal/v1/fridge-detection` | 냉장고 사진 물체 감지 |

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

---

## 디렉토리 구조

```
fresh-kitchen-ai-server/
├── main.py                     # FastAPI 서버 진입점
├── models/
│   ├── food_classifier/        # EfficientNet + Gemini 혼합 분류
│   ├── receipt_ocr/            # Document AI + Gemini OCR
│   └── object_detection/       # YOLOv8 물체 감지
├── training/                   # 모델 학습 스크립트
├── scripts/                    # 데이터 전처리 유틸리티 (Git 미포함)
├── yolo_model/                 # YOLOv8 모델 가중치 (Git 미포함)
├── receipt_model/              # Google Cloud 인증 파일 (Git 미포함)
├── picture_model/predict/      # 테스트용 이미지 샘플 (Git 미포함)
├── docs/                       # 개발 문서 및 학습 로그
├── .env.example
├── requirements.txt
└── README.md
```

---

## 보안 주의사항

다음 파일은 절대 Git에 올리지 마세요 (`.gitignore`에 자동 제외):

- `receipt-app-*.json` — Google Cloud 서비스 계정 키
- `.env` — API 키
- `*.pth`, `*.pt` — 모델 가중치
- `dataset/` — 학습 데이터
- `scripts/` — 데이터 전처리 스크립트

---

## 개발 환경

- Python 3.10.19 / PyTorch 2.5.1 / TorchVision 0.20.1
- FastAPI 0.135.1 / Uvicorn 0.42.0
- Google Cloud Document AI, Gemini 2.5 Flash
- Ultralytics YOLOv8

자세한 의존성은 `requirements.txt` 참조.
