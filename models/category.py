VALID_CATEGORIES = {"VEGETABLE", "FRUIT", "MEAT", "SEAFOOD", "DAIRY", "GRAIN", "SAUCE", "DRINK"}

def normalize_category(value: str) -> str:
    """Gemini가 예상 외 값을 반환할 때 ETC로 안전하게 fallback"""
    v = (value or "").strip().upper()
    return v if v in VALID_CATEGORIES else "ETC"
