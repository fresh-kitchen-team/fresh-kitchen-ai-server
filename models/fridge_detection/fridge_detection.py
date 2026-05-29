import os
import io
import json
import logging
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from models.category import normalize_category

MAX_IMAGE_PX = 1920  # Full HD — 냉장고 사진 인식에 충분, 고해상도 원본 업로드 방지

_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / '.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "60"))

logger = logging.getLogger(__name__)

# 모듈 수준 싱글톤 — 프로세스 전체에서 재사용
_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def detect_fridge_items(image_path: str) -> list:
    """냉장고 사진에서 Gemini Vision으로 식재료 목록 추출"""
    if not _gemini_client:
        logger.error("GEMINI_API_KEY가 .env에 없습니다.")
        return []

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_IMAGE_PX:
                scale = MAX_IMAGE_PX / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            image_bytes = buf.getvalue()
        mime_type = "image/jpeg"

        prompt = """
너는 냉장고 속 식재료를 파악하는 전문가야.
이 냉장고 사진을 보고 보이는 식재료와 식품을 모두 파악해줘.

[규칙]
- 식재료, 식품, 음료만 포함해
- 브랜드명은 제거하고 핵심 식품명만 써줘 (예: "풀무원 두부" → "두부")
- 반드시 한글로 출력해줘
- 뚜껑이나 포장재로 내용물이 안 보이는 것은 제외해줘
- "즉석식품", "음식", "채소", "소스", "식품" 처럼 포괄적인 단어는 쓰지 말고, 정확히 식별 안 되면 해당 항목을 제외해줘
- 같은 식재료는 중복 없이 하나만 써줘 (예: "계란"과 "달걀"은 "계란" 하나로, "탄산음료"와 "콜라"가 같으면 "콜라" 하나로)
- 각 식재료마다 카테고리 분류: VEGETABLE, FRUIT, MEAT, SEAFOOD, DAIRY, GRAIN, SAUCE, DRINK, ETC 중 하나
- 식재료가 없으면 빈 배열 []만 반환해줘

[출력 형식]
반드시 아래 JSON 배열 형식으로만 응답해. 다른 설명은 절대 쓰지 마.
[{"name": "재료1", "category": "VEGETABLE"}, {"name": "재료2", "category": "MEAT"}]
"""

        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                # SDK 네이티브 HTTP 타임아웃(ms) — 초과 시 요청 자체가 중단되어 스레드/워커 점유 없음
                http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT * 1000),
            )
        )

        items = json.loads(response.text)
        if not isinstance(items, list):
            return []

        # 중복 제거 + 형식 보정
        seen = set()
        result = []
        for item in items:
            if isinstance(item, dict):
                name = (item.get("name") or "").strip()
                category = normalize_category(item.get("category", "ETC"))
            else:
                name = str(item).strip()
                category = "ETC"
            if name and name not in seen:
                seen.add(name)
                result.append({"name": name, "category": category})
        return result

    except json.JSONDecodeError:
        logger.error("Gemini 응답 JSON 파싱 오류")
        return []
    except Exception as e:
        logger.error(f"Gemini 냉장고 탐지 오류: {e}")
        return []


if __name__ == "__main__":
    test_image = os.path.join(str(_PROJECT_ROOT), "samples", "fridge", "fridge_test.jpeg")
    print(f"🔍 '{test_image}' 분석 중...")
    result = detect_fridge_items(test_image)
    print(f"\n🥦 감지된 식재료 목록:")
    for item in result:
        print(f"  - {item['name']} ({item['category']})")
