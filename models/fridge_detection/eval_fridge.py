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

from models.eval_common import load_eval_samples, compare_items, prf, write_eval_csv
from models.fridge_detection.fridge_detection import detect_fridge_items

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(_BASE_DIR, "dataset", "eval", "fridge")
FIELDNAMES = ["사진", "품목", "상태", "정답_카테고리", "예측_카테고리"]


def main():
    samples = load_eval_samples(EVAL_DIR)
    if not samples:
        print(f"❌ 평가할 샘플이 없습니다: {EVAL_DIR}")
        print("   이미지와 같은 이름의 정답 .json 을 넣어주세요. (예: 도윤냉장1.jpeg + 도윤냉장1.json)")
        return

    print(f"🧊 냉장고 감지 평가 — 샘플 {len(samples)}장\n")

    rows = []
    tot_match = tot_miss = tot_extra = tot_cat = 0

    for img_path, gt in samples:
        name = os.path.basename(img_path)
        pred_items = detect_fridge_items(img_path)
        c = compare_items(pred_items, gt.get("ingredients", []))

        for r in c["rows"]:
            rows.append({"사진": name, **r})

        tot_match += c["match"]; tot_miss += c["miss"]; tot_extra += c["extra"]; tot_cat += c["cat_correct"]
        print(f"  {name:18s} 일치 {c['match']:2d} / 놓침 {c['miss']:2d} / 추가 {c['extra']:2d}")

    csv_path = write_eval_csv("fridge", rows, FIELDNAMES)

    # ---- 맨 아래 전체 수치 ----
    P, R, F1 = prf(tot_match, tot_extra, tot_miss)
    cat_acc = (tot_cat / tot_match * 100) if tot_match else 0.0

    print("\n" + "=" * 46)
    print("📊 전체 수치")
    print(f"   일치 {tot_match} / 놓침 {tot_miss} / 추가 {tot_extra}")
    print(f"   Precision {P*100:5.1f}%   Recall {R*100:5.1f}%   F1 {F1*100:5.1f}%")
    print(f"   카테고리 일치(일치 품목 중) {cat_acc:5.1f}% ({tot_cat}/{tot_match})")
    print("=" * 46)
    print(f"📝 비교 표 CSV: {os.path.relpath(csv_path, _BASE_DIR)}")


if __name__ == "__main__":
    main()
