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

def process_receipt_raw(file_path: str) -> list:
    """1차 정제: Document AI로 텍스트 추출 후 오직 한글만 보존"""
    if not os.path.exists(file_path):
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        return []

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

        return list(dict.fromkeys(raw_items))  # 기존 return raw_items 대신 이걸로!
    except Exception as e:
        logger.error(f"Document AI 처리 중 오류: {e}")
        return []

def filter_with_gemini(items: list) -> list:
    """2차 정제: Gemini 2.5 Flash를 사용하여 식재료만 추출 (JSON 강제)"""
    if not items:
        logger.warning("1차 필터링 결과가 비어있어 Gemini 호출을 건너뜁니다.")
        return []

    logger.info("[2/2] Gemini LLM 2차 정제 중 (식재료만 추출)...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    items_str = "\n".join(f"- {item}" for item in items)

    prompt = f"""
너는 영수증 텍스트 정제 전문가야. 아래는 영수증에서 추출한 텍스트 목록이야.

[할 일]
이 중에서 실제 식재료, 식품, 음료만 골라내줘.

[제외 항목]
- 카드명, 포인트, 환불, 할인, 과세매출, 합계, 부가세, 매장명
- 봉투, 비닐백 등 포장재/서비스
- 의미를 알 수 없는 파편화된 단어

[변환 규칙]
- 브랜드명은 제거하고 핵심 식품명만 남겨줘. (예: "CJ 비비고 만두" → "만두", "풀무원 두부" → "두부")
- 반드시 한글로 출력해줘.
- 중복된 항목은 하나만 남겨줘.

[출력 형식]
반드시 아래 JSON 배열 형식으로만 응답해. 다른 설명은 절대 쓰지 마.
["재료1", "재료2", "재료3"]

[데이터]
{items_str}
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
                return []

        final_list = json.loads(response.text)
        if not isinstance(final_list, list):
            logger.error(f"예상치 못한 JSON 형식: {final_list}")
            return []
        return final_list

    except json.JSONDecodeError:
        logger.error("Gemini 응답 JSON 파싱 오류")
        return []
    except Exception as e:
        logger.error(f"Gemini 오류: {e}")
        return []

if __name__ == "__main__":
    TARGET_FILE = os.path.join(str(_PROJECT_ROOT), 'test_real_image', 'receipt.png')
    
    raw_list = process_receipt_raw(TARGET_FILE)
    
    if raw_list:
        print(f"\n🔍 [1차 필터링 결과] (한글만 추출):")
        print(raw_list)
        
        final_list = filter_with_gemini(raw_list)
        
        print(f"\n✨ [최종 식재료 리스트] ✨\n" + "-"*30)
        if isinstance(final_list, list) and final_list:
            for food in final_list: 
                print(f"📍 {food}")
        else:
            print("추출된 식재료가 없거나 파싱에 실패했습니다.")
        print("-" * 30)
    else:
        print("영수증에서 유효한 글자를 찾지 못했습니다.")