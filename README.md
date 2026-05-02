# fresh-kitchen-ai-server

fresh-kitchen 프로젝트의 AI 서버 — 음식 이미지 분류, 영수증 OCR, 냉장고 물체 감지를 담당합니다.

## 구성

### models/food_classifier — 음식 이미지 분류
- **모델**: EfficientNet V2-M
- 확신도 75% 미만이면 Gemini Vision으로 자동 위임
- `predict_V2-M.py`: 이미지 예측
- `test_V2-M.py`: 모델 정확도 평가

### models/receipt_ocr — 영수증 OCR
- **기술**: Google Cloud Document AI + Gemini API
- 영수증 이미지 → 텍스트 추출 → 식재료 목록 반환
- `receipt_ocr.py`

### models/object_detection — 냉장고 물체 감지
- **모델**: YOLOv8n
- 냉장고/음식 물품 탐지
- `yolo.py`, `yolo_predict.py`

### training — 모델 학습
- `train_EfficientNet_V2-M.py`: EfficientNet V2-M 학습

### scripts — 데이터 전처리 유틸리티
| 스크립트 | 역할 |
|----------|------|
| `data_crawl.py` | Bing에서 음식 이미지 크롤링 |
| `data_clean.py` | 중복/손상 이미지 제거 |
| `data_split.py` | train/val/test 분할 |
| `data_diet.py` | 클래스 불균형 조정 |
| `data_blurred.py` | 블러 처리 검증 데이터 생성 |
| `data_len.py` | 데이터셋 현황 리포트 |

---

## 설치

```bash
# Python 3.10.19 필수
conda activate project
pip install -r requirements.txt
```

```bash
cp .env.example .env
# .env 파일에 자신의 API 키 입력
```

모델 가중치(`best_food_model_v2_m.pth`)는 Git에 포함되지 않습니다. 팀 드라이브에서 다운로드 후 프로젝트 루트에 저장하세요.

---

## 실행

```bash
# 음식 이미지 분류
python models/food_classifier/predict_V2-M.py

# 영수증 분석
python models/receipt_ocr/receipt_ocr.py

# 냉장고 물체 감지
python models/object_detection/yolo_predict.py
```

---

## 디렉토리 구조

```
fresh-kitchen-ai-server/
├── models/
│   ├── food_classifier/        # EfficientNet + Gemini 혼합 분류
│   ├── receipt_ocr/            # Document AI + Gemini OCR
│   └── object_detection/       # YOLOv8 물체 감지
├── training/                   # 모델 학습 스크립트
├── scripts/                    # 데이터 전처리 유틸리티
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

---

## 개발 환경

- Python 3.10.19 / PyTorch 2.5.1 / TorchVision 0.20.1
- Google Cloud Document AI, Gemini 2.5 Flash
- Ultralytics YOLOv8, OpenCV, Pillow

자세한 의존성은 `requirements.txt` 참조.
