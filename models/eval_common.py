"""영수증 OCR · 냉장고 감지 평가 공용 유틸리티.

두 모델 모두 자유 표기의 품목 리스트를 출력하므로, 단일 점수(F1)로 깎기보다
사진별 '예측 vs 정답'을 품목 단위로 대조해 일치/놓침/추가로 가른다.
품목 이름은 공백만 무시한 정확 일치(동의어 보정 없음 — 보이는 것을 그대로 뽑는지를 정직하게 본다).
"""

import os
import csv
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# 평가 결과 CSV가 쌓이는 곳 — 종류별 폴더 docs/logs/{fridge,receipt}/
# (분류모델은 docs/logs/classifier/verX 계열)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_BASE_DIR = os.path.join(_BASE_DIR, "docs", "logs")


def normalize_name(name: str) -> str:
    """품목 이름 정규화: 공백만 제거(예: '방울 토마토' == '방울토마토'). 동의어 보정 없음."""
    return (name or "").strip().replace(" ", "")


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
# 정답 vs 예측 비교
# ==========================================
def _name_to_raw(items: list) -> dict:
    """[{name, ...}] -> {정규화된 이름: 원본 이름} (먼저 나온 항목 우선)."""
    mapping = {}
    for it in items or []:
        raw = (it.get("name") if isinstance(it, dict) else str(it)) or ""
        raw = raw.strip()
        key = normalize_name(raw)
        if key:
            mapping.setdefault(key, raw)
    return mapping


def compare_items(pred_items: list, gt_items: list) -> dict:
    """한 사진의 정답 vs 예측을 품목별로 비교.

    반환 rows: [{"품목","상태"}, ...]
      - 상태: 일치(정답·예측 모두 있음) / 놓침(정답에만) / 추가(예측에만)
    counts: 일치·놓침·추가 수
    """
    pred = _name_to_raw(pred_items)
    gt = _name_to_raw(gt_items)

    rows = []
    match = miss = extra = 0

    # 정답 순서대로: 일치 / 놓침
    for key, raw in gt.items():
        if key in pred:
            rows.append({"품목": raw, "상태": "일치"})
            match += 1
        else:
            rows.append({"품목": raw, "상태": "놓침"})
            miss += 1

    # 예측에만 있는 것: 추가(오검출)
    for key, raw in pred.items():
        if key not in gt:
            rows.append({"품목": raw, "상태": "추가"})
            extra += 1

    return {"rows": rows, "match": match, "miss": miss, "extra": extra}


def write_eval_csv(kind: str, rows: list, fieldnames: list) -> str:
    """평가 결과를 docs/logs/{kind}/{kind}_{타임스탬프}.csv 로 저장하고 경로 반환.

    kind 별 폴더(fridge/·receipt/)에 종류를 나눠 쌓는다.
    실행할 때마다 새 파일이 생겨 과거 결과를 덮어쓰지 않는다.
    분류모델 class_metrics_val.csv 와 동일하게 utf-8-sig(BOM) — 엑셀 한글 깨짐 방지.
    """
    out_dir = os.path.join(LOG_BASE_DIR, kind)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{kind}_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
