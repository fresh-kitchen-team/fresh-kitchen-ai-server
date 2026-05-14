import os
import re
import json
import logging
import mimetypes
import concurrent.futures
from google.cloud import documentai
from google import genai
from google.genai import types

# ==========================================
# [1. 설정 구역] 
# ==========================================
from dotenv import load_dotenv
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / '.env')

# GOOGLE_APPLICATION_CREDENTIALS: None 방지 + 상대경로 → 절대경로 변환
_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if _creds:
    if not os.path.isabs(_creds):
        _creds = str((_PROJECT_ROOT / _creds).resolve())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _creds

project_id = os.getenv("PROJECT_ID")
processor_id = os.getenv("PROCESSOR_ID")
location = os.getenv("LOCATION")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TIMEOUT = 30  # seconds

logger = logging.getLogger(__name__)

# ==========================================

def process_receipt_raw(file_path: str) -> dict:
    """1차 정제: Document AI로 텍스트 추출 후 한글 품목 + 전체 OCR 텍스트 반환"""
    if not os.path.exists(file_path):
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        return {}

    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project_id, location, processor_id)

    try:
        with open(file_path, "rb") as image:
            image_content = image.read()

        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=name, raw_document=raw_document)

        logger.info("[1/2] Document AI 스캔 및 1차 정제 중...")
        result = client.process_document(request=request)

        # 전체 OCR 텍스트
        ocr_text = result.document.text or ""

        # 한글 품목만 추출
        raw_items = []
        for entity in result.document.entities:
            if entity.type_ == "line_item":
                for prop in entity.properties:
                    if prop.type_ == "line_item/description":
                        text = prop.mention_text.strip()
                        clean_text = re.sub(r'[^가-힣\s]', '', text).strip()
                        clean_text = re.sub(r'\s+', ' ', clean_text)
                        if clean_text and len(clean_text) > 1:
                            raw_items.append(clean_text)

        return {
            "raw_items": list(dict.fromkeys(raw_items)),
            "ocr_text": ocr_text,
        }
    except Exception as e:
        logger.error(f"Document AI 처리 중 오류: {e}")
        return {}


def filter_with_gemini(raw_data: dict) -> dict:
    """2차 정제: Gemini 2.5 Flash로 날짜·식재료 추출 (JSON 강제)"""
    raw_items = raw_data.get("raw_items", [])
    ocr_text = raw_data.get("ocr_text", "")  # Gemini 프롬프트용 (응답에는 미포함)

    if not raw_items and not ocr_text:
        logger.warning("1차 필터링 결과가 비어있어 Gemini 호출을 건너뜁니다.")
        return {"purchasedAt": None, "ingredients": []}

    logger.info("[2/2] Gemini LLM 2차 정제 중 (날짜·식재료 추출)...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    items_str = "\n".join(f"- {item}" for item in raw_items) if raw_items else "(품목 목록 없음 — 전체 텍스트에서 직접 추출)"

    prompt = f"""
너는 영수증 분석 전문가야. 아래 영수증 전체 텍스트와 품목 목록을 보고 JSON으로 답해줘.

[영수증 전체 텍스트]
{ocr_text}

[추출된 품목 목록]
{items_str}

[할 일]
1. 구매일(purchasedAt): 날짜를 YYYY-MM-DD 형식으로 추출해. 없으면 null.
2. 식재료(ingredients): 품목 목록이 있으면 거기서, 없으면 전체 텍스트에서 실제 식재료·식품·음료만 골라내줘.
   - 브랜드명 제거 (예: "CJ 비비고 만두" → "만두", "풀무원 두부" → "두부")
   - 카드명, 포인트, 환불, 할인, 합계, 부가세, 봉투 등 비식재료 제외
   - 중복 제거, 반드시 한글로 출력

[출력 형식]
반드시 아래 JSON 형식으로만 응답해. 다른 설명은 절대 쓰지 마.
{{"purchasedAt": "YYYY-MM-DD 또는 null", "ingredients": ["재료1", "재료2"]}}
"""

    def _call():
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            try:
                response = future.result(timeout=GEMINI_TIMEOUT)
            except concurrent.futures.TimeoutError:
                logger.error(f"Gemini API 타임아웃 ({GEMINI_TIMEOUT}초 초과)")
                return {"purchasedAt": None, "ingredients": []}

        parsed = json.loads(response.text)
        if not isinstance(parsed, dict):
            logger.error(f"예상치 못한 JSON 형식: {parsed}")
            return {"purchasedAt": None, "ingredients": []}

        return {
            "purchasedAt": parsed.get("purchasedAt"),
            "ingredients": parsed.get("ingredients", []),
        }

    except json.JSONDecodeError:
        logger.error("Gemini 응답 JSON 파싱 오류")
        return {"purchasedAt": None, "ingredients": []}
    except Exception as e:
        logger.error(f"Gemini 오류: {e}")
        return {"purchasedAt": None, "ingredients": []}


if __name__ == "__main__":
    TARGET_FILE = os.path.join(str(_PROJECT_ROOT), 'dataset', 'test_real_image', 'receipt.png')

    raw_data = process_receipt_raw(TARGET_FILE)

    if raw_data:
        print(f"\n🔍 [1차 필터링 결과] (한글만 추출):")
        print(raw_data["raw_items"])

        result = filter_with_gemini(raw_data)

        print(f"\n✨ [최종 분석 결과] ✨\n" + "-"*30)
        print(f"📅 구매일: {result['purchasedAt']}")
        print(f"🥦 식재료: {result['ingredients']}")
        print("-" * 30)
    else:
        print("영수증에서 유효한 글자를 찾지 못했습니다.")