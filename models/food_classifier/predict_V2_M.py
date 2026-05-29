import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import mimetypes
import os
import shutil
import json
from datetime import datetime
import logging
import concurrent.futures
from dotenv import load_dotenv
from google import genai
from google.genai import types
from models.category import normalize_category


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_BASE_DIR, '.env'))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CONFIDENCE_THRESHOLD = 80.0
SAVE_DIR = os.path.join(_BASE_DIR, 'dataset', 'auto_labeled')
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "60"))

logger = logging.getLogger(__name__)

CLASS_CATEGORY = {
    "멸치": "SEAFOOD",
    "사과": "FRUIT",
    "베이컨": "MEAT",
    "바나나": "FRUIT",
    "콩나물": "VEGETABLE",
    "소고기": "MEAT",
    "맥주": "DRINK",
    "파프리카": "VEGETABLE",
    "블루베리": "FRUIT",
    "브로콜리": "VEGETABLE",
    "버터": "DAIRY",
    "양배추": "VEGETABLE",
    "당근": "VEGETABLE",
    "치즈": "DAIRY",
    "닭고기": "MEAT",
    "고추": "VEGETABLE",
    "고춧가루": "SAUCE",
    "부추": "VEGETABLE",
    "콜라": "DRINK",
    "식용유": "ETC",
    "옥수수": "GRAIN",
    "오이": "VEGETABLE",
    "된장": "SAUCE",
    "계란": "ETC",
    "가지": "VEGETABLE",
    "마늘": "VEGETABLE",
    "당면": "GRAIN",
    "고추장": "SAUCE",
    "대파": "VEGETABLE",
    "다시마": "SEAFOOD",
    "케첩": "SAUCE",
    "김치": "VEGETABLE",
    "키위": "FRUIT",
    "상추": "VEGETABLE",
    "고등어": "SEAFOOD",
    "마요네즈": "SAUCE",
    "우유": "DAIRY",
    "버섯": "VEGETABLE",
    "머스타드": "SAUCE",
    "양파": "VEGETABLE",
    "오렌지": "FRUIT",
    "굴소스": "SAUCE",
    "배": "FRUIT",
    "후추": "ETC",
    "깻잎": "VEGETABLE",
    "돼지고기": "MEAT",
    "감자": "VEGETABLE",
    "호박": "VEGETABLE",
    "무": "VEGETABLE",
    "라면사리": "GRAIN",
    "떡": "GRAIN",
    "소금": "ETC",
    "햄": "MEAT",
    "미역": "SEAFOOD",
    "참깨": "GRAIN",
    "참기름": "SAUCE",
    "새우": "SEAFOOD",
    "소주": "DRINK",
    "간장": "SAUCE",
    "시금치": "VEGETABLE",
    "오징어": "SEAFOOD",
    "쌈장": "SAUCE",
    "딸기": "FRUIT",
    "설탕": "ETC",
    "고구마": "VEGETABLE",
    "두부": "ETC",
    "토마토": "VEGETABLE",
    "참치": "SEAFOOD",
    "식초": "SAUCE",
    "애호박": "VEGETABLE",
}

_TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(480),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 모듈 수준 싱글톤 — 프로세스 전체에서 재사용
_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# --------------------------------------
# 1. 모델 준비 함수 (서버 켜질 때 딱 1번만 실행됨)
# --------------------------------------
def load_food_model(model_path: str):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"AI 장치 설정: {device}")

    try:
        model = models.efficientnet_v2_m(weights=None)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        class_names = checkpoint['class_names']

        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(num_ftrs, len(class_names))
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logger.info("식재료 인식 모델이 메모리에 로드되었습니다.")

        # CLASS_CATEGORY 누락 감지 — 신규 클래스가 추가되면 사일런트 "ETC" 매핑을 막기 위해 경고
        missing = [c for c in class_names if c not in CLASS_CATEGORY]
        if missing:
            logger.warning(f"CLASS_CATEGORY에 누락된 클래스 {len(missing)}개: {missing} → ETC로 처리됩니다.")

        return model, device, class_names

    except Exception as e:
        logger.error(f"모델 초기화 오류: {e}")
        return None, None, None

# --------------------------------------
# 2. Gemini Vision 예측 함수
# --------------------------------------
def gemini_predict(image_path: str, class_names: list) -> dict:
    """EfficientNet이 확신 못할 때 Gemini Vision이 대신 판단"""
    if not _gemini_client:
        return {"error": "GEMINI_API_KEY가 .env에 없습니다."}

    try:
        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        class_list_str = "\n".join(f"- {c}" for c in class_names)
        prompt = f"""
너는 식재료 이미지 분류 전문가야.
아래 사진 속 식재료를 파악하고, 아래 규칙에 따라 JSON으로만 답해줘.

[클래스 목록]
{class_list_str}

[규칙]
- 사진 속 식재료가 클래스 목록에 있으면 반드시 목록의 클래스명 그대로 반환해줘.
- 클래스 목록에 없는 식재료면 한국어 식품명(예: 수박, 망고)으로 반환해줘.
- category는 반드시 아래 중 하나로 분류해: VEGETABLE, FRUIT, MEAT, SEAFOOD, DAIRY, GRAIN, SAUCE, DRINK, ETC

[출력 형식] 다른 설명 없이 아래 JSON만 출력:
{{"label": "클래스명 또는 한국어 식품명", "category": "카테고리"}}
"""

        def _call():
            return _gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                )
            )

        future = _executor.submit(_call)
        try:
            response = future.result(timeout=GEMINI_TIMEOUT)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.error(f"Gemini API 타임아웃 ({GEMINI_TIMEOUT}초 초과)")
            return {"error": "Gemini 타임아웃"}

        result = json.loads(response.text)
        # Gemini 2.5가 종종 단일 객체를 [{...}] 배열로 감싸서 반환 → 첫 원소 추출
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            return {"error": "Gemini 응답 형식 오류"}
        return {
            "label": result.get("label", "unknown"),
            "category": result.get("category", "ETC"),
        }

    except json.JSONDecodeError:
        return {"error": "Gemini 응답 파싱 실패"}
    except Exception as e:
        logger.error(f"Gemini 오류: {e}")
        return {"error": f"Gemini 오류: {e}"}

# --------------------------------------
# 3. 자동 저장 함수 (Self-improving 핵심)
# --------------------------------------
def save_to_dataset(image_path: str, class_name: str):
    """Gemini가 분류한 이미지를 auto_labeled에 자동 저장"""
    try:
        save_folder = os.path.join(SAVE_DIR, class_name)
        os.makedirs(save_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ext = os.path.splitext(image_path)[1] or ".jpg"
        save_path = os.path.join(save_folder, f"auto_{timestamp}{ext}")

        shutil.copy2(image_path, save_path)
        logger.info(f"자동 저장: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"자동 저장 실패: {e}")
        return None

# --------------------------------------
# 4. 메인 예측 함수
# --------------------------------------
def predict_image(model, device, image_path: str, class_names: list) -> dict:
    if model is None:
        return {"error": "모델이 로드되지 않았습니다."}

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = _TRANSFORM(image).unsqueeze(0).to(device)

    except FileNotFoundError:
        return {"error": f"'{image_path}' 사진을 찾을 수 없습니다."}

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

        conf, predicted_idx = torch.max(probabilities, 0)
        best_class = class_names[predicted_idx.item()]
        confidence = round(conf.item(), 2)

        top3_prob, top3_idx = torch.topk(probabilities, 3)
        top3_list = [
            {"name": class_names[top3_idx[i].item()], "confidence": round(top3_prob[i].item(), 2)}
            for i in range(3)
        ]

    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(f"확신도 낮음 ({confidence}%) → Gemini 폴백")
        gemini_result = gemini_predict(image_path, class_names)

        if "error" in gemini_result:
            logger.warning(f"Gemini 실패: {gemini_result['error']} → EfficientNet 결과 사용")
            return {
                "best_match": best_class,
                "category": normalize_category(CLASS_CATEGORY.get(best_class, "ETC")),
                "confidence": confidence,
                "top3": top3_list,
                "source": "efficientnet_fallback"
            }

        gemini_label = gemini_result["label"]
        saved_path = save_to_dataset(image_path, gemini_label)

        return {
            "best_match": gemini_label,
            "category": normalize_category(gemini_result.get("category", "ETC")),
            "confidence": confidence,
            "top3": [],
            "source": "gemini",
            "auto_saved": saved_path
        }

    return {
        "best_match": best_class,
        "category": normalize_category(CLASS_CATEGORY.get(best_class, "ETC")),
        "confidence": confidence,
        "top3": top3_list,
        "source": "efficientnet"
    }

# ==========================================
# [Main 블록] 단독 실행 테스트용
# ==========================================
if __name__ == '__main__':
    _MODEL_PATH = os.path.join(_BASE_DIR, 'best_food_model_v2_m_ver5.pth')
    TEST_IMAGE_PATH = os.path.join(_BASE_DIR, 'samples', 'food', 'beef1.jpeg')

    my_model, my_device, class_names = load_food_model(_MODEL_PATH)

    if my_model:
        print(f"\n🔍 '{TEST_IMAGE_PATH}' 분석 중...")
        result = predict_image(my_model, my_device, TEST_IMAGE_PATH, class_names)

        if "error" in result:
            print(result["error"])
        else:
            print("\n" + "="*35)
            print(f"🍎 예측 결과: [{result['best_match']}]")
            print(f"📊 확신도: {result['confidence']}%")
            print(f"🔧 판단 주체: {result['source']}")
            if result.get("auto_saved"):
                print(f"💾 자동 저장됨: {result['auto_saved']}")
            print("="*35)
            print("\n[Top 3 후보]")
            for i, item in enumerate(result['top3']):
                print(f"  {i+1}위: {item['name']} ({item['confidence']}%)")
