VALID_CATEGORIES = {"VEGETABLE", "FRUIT", "MEAT", "SEAFOOD", "DAIRY", "GRAIN", "SAUCE", "DRINK"}

def normalize_category(value: str) -> str:
    v = (value or "").strip().upper()
    return v if v in VALID_CATEGORIES else "ETC"
