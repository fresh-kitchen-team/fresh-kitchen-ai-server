"""냉장고 감지 평가 — 정답 vs 예측을 품목별 표(CSV)로 비교.

dataset/eval/fridge/ 안의 냉장고 사진마다 같은 이름의 정답 JSON을 두고,
detect_fridge_items() 예측을 정답과 품목별로 비교해 CSV 표로 저장한다.
CSV 열: 사진 | 품목 | 상태(일치/놓침/추가) | 정답_카테고리 | 예측_카테고리

정답 JSON 형식 (예: fridge1.jpeg → fridge1.json):
    {"ingredients": [{"name": "두부", "category": "GRAIN"}, ...]}

실행:
    python -m models.fridge_detection.eval_fridge
"""

import os

from models.eval_common import load_eval_samples, compare_items, write_eval_csv
from models.fridge_detection.fridge_detection import detect_fridge_items

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(_BASE_DIR, "dataset", "eval", "fridge")
FIELDNAMES = ["사진", "예측", "정답", "놓침", "추가"]


def _names(items):
    return ", ".join((it.get("name") or "").strip() for it in items) or "(없음)"


def main():
    samples = load_eval_samples(EVAL_DIR)
    if not samples:
        print(f"❌ 평가할 샘플이 없습니다: {EVAL_DIR}")
        print("   이미지와 같은 이름의 정답 .json 을 넣어주세요. (예: 도윤냉장1.jpeg + 도윤냉장1.json)")
        return

    print(f"🧊 냉장고 감지 평가 — 샘플 {len(samples)}장\n")

    rows = []
    tot_match = tot_miss = tot_extra = 0

    for img_path, gt in samples:
        name = os.path.basename(img_path)
        gt_items = gt.get("ingredients", [])
        pred_items = detect_fridge_items(img_path)
        c = compare_items(pred_items, gt_items)

        missed = [r["품목"] for r in c["rows"] if r["상태"] == "놓침"]
        extra = [r["품목"] for r in c["rows"] if r["상태"] == "추가"]
        rows.append({
            "사진": name,
            "예측": _names(pred_items),   # 모델이 뽑은 것
            "정답": _names(gt_items),     # 실제 있던 것
            "놓침": ", ".join(missed),    # 정답엔 있는데 모델이 못 잡음
            "추가": ", ".join(extra),     # 모델이 잡았는데 실제론 없음(오검출)
        })

        tot_match += c["match"]; tot_miss += c["miss"]; tot_extra += c["extra"]
        print(f"  {name:18s} 일치 {c['match']:2d} / 놓침 {c['miss']:2d} / 추가 {c['extra']:2d}")

    csv_path = write_eval_csv("fridge", rows, FIELDNAMES)

    # ---- 맨 아래 합계 (표기 차이 포함된 거친 수치 — 자세한 건 비교표 참고) ----
    print("\n" + "=" * 46)
    print(f"📊 합계  일치 {tot_match} / 놓침 {tot_miss} / 추가 {tot_extra}")
    print("   (놓침·추가엔 표기만 다른 같은 품목도 섞임 — 판단은 비교표로)")
    print("=" * 46)
    print(f"📝 비교 표 CSV: {os.path.relpath(csv_path, _BASE_DIR)}")


if __name__ == "__main__":
    main()
