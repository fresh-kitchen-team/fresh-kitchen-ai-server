"""영수증 OCR · 냉장고 감지 정확도 평가 공용 유틸리티.

두 모델 모두 '품목 리스트 + 카테고리'를 출력하므로, 단일 라벨 정확도가 아니라
집합 기반 Precision / Recall / F1 로 평가한다. 품목 이름은 정확 일치를 기본으로 하되
동의어 테이블(SYNONYMS)로 표기 차이를 흡수한다.
"""

import os
import csv
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# 평가 결과 CSV가 쌓이는 곳 (분류모델 docs/logs/verX 와 같은 계열)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_LOG_DIR = os.path.join(_BASE_DIR, "docs", "logs", "eval")

# ==========================================
# 동의어 테이블
# ==========================================
# canonical(대표 표기) -> 같은 의미로 취급할 표기들.
# 예측·정답 양쪽 모두 이 테이블로 대표 표기에 매핑한 뒤 비교한다.
# 새 동의어는 여기에 줄만 추가하면 평가에 즉시 반영된다.
SYNONYMS = {
    "계란": ["달걀", "에그", "유정란"],
    "대파": ["파"],
    "콜라": ["코카콜라", "펩시", "탄산음료"],
    "사이다": ["스프라이트"],
    "두부": ["연두부", "순두부"],
    "삼겹살": ["대패삼겹", "냉동삼겹"],
    "돼지고기": ["돈육", "포크"],
    "소고기": ["쇠고기", "우육", "비프"],
    "닭고기": ["닭", "치킨"],
    "고구마": ["군고구마"],
    "감자": ["포테이토"],
    "토마토": ["방울토마토", "대추토마토"],
    "양파": ["적양파"],
    "우유": ["흰우유", "멸균우유"],
    "치즈": ["슬라이스치즈"],
    "오징어": ["물오징어", "갑오징어"],
    "새우": ["흰다리새우", "대하"],
    "당근": ["홍당무"],
}

# alias -> canonical 역방향 조회 테이블 (대표 표기 자신도 포함)
_ALIAS_TO_CANON = {}
for _canon, _aliases in SYNONYMS.items():
    _ALIAS_TO_CANON[_canon.replace(" ", "")] = _canon
    for _a in _aliases:
        _ALIAS_TO_CANON[_a.replace(" ", "")] = _canon


def normalize_name(name: str) -> str:
    """품목 이름 정규화: 공백 제거 후 동의어를 대표 표기로 환산."""
    key = (name or "").strip().replace(" ", "")
    return _ALIAS_TO_CANON.get(key, key)


# ==========================================
# 정답(ground truth) 로딩
# ==========================================
def load_eval_samples(eval_dir: str) -> list:
    """eval_dir 안의 이미지마다 같은 이름의 .json 정답을 짝지어 반환.

    반환: [(image_path, gt_dict), ...]  — 정답 JSON이 없는 이미지는 건너뛴다.
    gt_dict 예시: {"purchasedAt": "2026-05-13",
                   "ingredients": [{"name": "두부", "category": "GRAIN"}, ...]}
    """
    if not os.path.isdir(eval_dir):
        logger.error(f"평가 폴더가 없습니다: {eval_dir}")
        return []

    samples = []
    for fname in sorted(os.listdir(eval_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTS:
            continue
        img_path = os.path.join(eval_dir, fname)
        gt_path = os.path.join(eval_dir, stem + ".json")
        if not os.path.exists(gt_path):
            logger.warning(f"정답 JSON 없음 — 건너뜀: {fname}")
            continue
        try:
            with open(gt_path, encoding="utf-8") as f:
                gt = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"정답 JSON 파싱 실패 — 건너뜀: {gt_path} ({e})")
            continue
        samples.append((img_path, gt))
    return samples


# ==========================================
# 집합 기반 채점
# ==========================================
def _name_to_category(items: list) -> dict:
    """[{name, category}] -> {정규화된 이름: 카테고리} (먼저 나온 항목 우선)."""
    mapping = {}
    for it in items or []:
        if not isinstance(it, dict):
            name = normalize_name(str(it))
            mapping.setdefault(name, "ETC")
            continue
        name = normalize_name(it.get("name", ""))
        if not name:
            continue
        mapping.setdefault(name, (it.get("category") or "ETC"))
    return mapping


def score_items(pred_items: list, gt_items: list) -> dict:
    """한 이미지의 품목 리스트를 채점.

    - 이름 매칭: 정규화(동의어 흡수) 후 집합 교집합으로 TP 판정
    - 카테고리: 이름이 맞은(TP) 항목 중 카테고리까지 일치한 비율
    """
    pred = _name_to_category(pred_items)
    gt = _name_to_category(gt_items)

    pred_names = set(pred)
    gt_names = set(gt)

    matched = pred_names & gt_names
    tp = len(matched)
    fp = len(pred_names - gt_names)
    fn = len(gt_names - pred_names)

    cat_correct = sum(1 for n in matched if pred[n] == gt[n])

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "cat_correct": cat_correct,
        "matched": matched,
        "missed": gt_names - pred_names,    # 정답엔 있으나 예측 못 한 것 (FN)
        "extra": pred_names - gt_names,     # 예측했으나 정답에 없는 것 (FP)
        "pred": pred, "gt": gt,
    }


def prf(tp: int, fp: int, fn: int) -> tuple:
    """(precision, recall, f1) — 분모 0 방어."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def write_eval_csv(kind: str, rows: list, fieldnames: list) -> str:
    """평가 결과를 docs/logs/eval/{kind}_{타임스탬프}.csv 로 저장하고 경로 반환.

    실행할 때마다 새 파일이 생겨 과거 결과를 덮어쓰지 않는다.
    분류모델 class_metrics_val.csv 와 동일하게 utf-8-sig(BOM) — 엑셀 한글 깨짐 방지.
    """
    os.makedirs(EVAL_LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EVAL_LOG_DIR, f"{kind}_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
