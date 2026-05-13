import os
import json
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / '.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def detect_fridge_items(image_path: str) -> list:
    """냉장고 사진에서 Gemini Vision으로 식재료 목록 추출"""
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY가 .env에 없습니다.")
        return []

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"

        prompt = """
너는 냉장고 속 식재료를 파악하는 전문가야.
이 냉장고 사진을 보고 보이는 식재료와 식품 재료를 모두 파악해줘.

[규칙]
- 식재료, 식품, 음료만 포함해
- 브랜드명은 제거하고 핵심 식품명만 써줘 (예: "풀무원 두부" → "두부")
- 반드시 한글로 출력해줘
- 확실하지 않은 것은 제외해줘

[출력 형식]
반드시 아래 JSON 배열 형식으로만 응답해. 다른 설명은 절대 쓰지 마.
["재료1", "재료2", "재료3"]
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )

        items = json.loads(response.text)
        if not isinstance(items, list):
            return []
        return items

    except json.JSONDecodeError:
        print("❌ JSON 파싱 오류")
        return []
    except Exception as e:
        print(f"❌ Gemini 냉장고 탐지 오류: {e}")
        return []


if __name__ == "__main__":
    test_image = os.path.join(str(_PROJECT_ROOT), "picture_model", "predict", "fridge_test.jpeg")
    print(f"🔍 '{test_image}' 분석 중...")
    result = detect_fridge_items(test_image)
    print(f"\n🥦 감지된 식재료 목록:")
    for item in result:
        print(f"  - {item}")
