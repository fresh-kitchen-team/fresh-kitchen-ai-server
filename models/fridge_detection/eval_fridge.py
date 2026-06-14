"""냉장고 감지 정확도 평가.

dataset/eval/fridge/ 안의 냉장고 사진마다 같은 이름의 정답 JSON을 두고,
detect_fridge_items() 예측과 비교해 품목 Precision/Recall/F1 과 카테고리 정확도를 낸다.

정답 JSON 형식 (예: fridge1.jpeg → fridge1.json):
    {"ingredients": [{"name": "두부", "category": "GRAIN"},
                     {"name": "계란", "category": "DAIRY"}]}

실행:
    python -m models.fridge_detection.eval_fridge
"""

import os

from models.eval_common import load_eval_samples, score_items, prf, write_eval_csv
from models.fridge_detection.fridge_detection import detect_fridge_items

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(_BASE_DIR, "dataset", "eval", "fridge")


def main():
    samples = load_eval_samples(EVAL_DIR)
    if not samples:
        print(f"❌ 평가할 샘플이 없습니다: {EVAL_DIR}")
        print("   이미지와 같은 이름의 정답 .json 을 넣어주세요. (예: 도윤냉장1.jpeg + 도윤냉장1.json)")
        return

    print(f"🧊 냉장고 감지 평가 시작 — 샘플 {len(samples)}장\n")

    tot_tp = tot_fp = tot_fn = tot_cat = 0
    rows = []  # CSV 행 누적

    for img_path, gt in samples:
        name = os.path.basename(img_path)
        pred_items = detect_fridge_items(img_path)
        s = score_items(pred_items, gt.get("ingredients", []))

        tot_tp += s["tp"]; tot_fp += s["fp"]; tot_fn += s["fn"]; tot_cat += s["cat_correct"]
        p, r, f1 = prf(s["tp"], s["fp"], s["fn"])

        rows.append({
            "image": name,
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "cat_correct": s["cat_correct"],
            "cat_acc": round(s["cat_correct"] / s["tp"], 4) if s["tp"] else "",
            "missed": ";".join(sorted(s["missed"])),
            "extra": ";".join(sorted(s["extra"])),
        })

        print(f"📷 {name}")
        print(f"   P {p*100:5.1f}%  R {r*100:5.1f}%  F1 {f1*100:5.1f}%  "
              f"(TP {s['tp']} / FP {s['fp']} / FN {s['fn']})")
        if s["missed"]:
            print(f"   🔴 놓침(FN): {', '.join(sorted(s['missed']))}")
        if s["extra"]:
            print(f"   🟡 오검출(FP): {', '.join(sorted(s['extra']))}")
        if s["tp"]:
            cat_acc = s["cat_correct"] / s["tp"] * 100
            print(f"   🏷  카테고리 정확도(맞춘 품목 기준): {cat_acc:5.1f}% ({s['cat_correct']}/{s['tp']})")
        print()

    # --------------------------------------
    # 전체 집계 (micro 평균)
    # --------------------------------------
    P, R, F1 = prf(tot_tp, tot_fp, tot_fn)
    cat_acc = (tot_cat / tot_tp * 100) if tot_tp else 0.0

    print("=" * 50)
    print("🏆 전체 집계 (micro 평균)")
    print(f"   Precision : {P*100:5.1f}%")
    print(f"   Recall    : {R*100:5.1f}%")
    print(f"   F1        : {F1*100:5.1f}%")
    print(f"   품목 합계 : TP {tot_tp} / FP {tot_fp} / FN {tot_fn}")
    print(f"   카테고리 정확도(맞춘 품목 기준): {cat_acc:5.1f}% ({tot_cat}/{tot_tp})")
    print("=" * 50)

    rows.append({
        "image": "__TOTAL__",
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "precision": round(P, 4), "recall": round(R, 4), "f1": round(F1, 4),
        "cat_correct": tot_cat,
        "cat_acc": round(tot_cat / tot_tp, 4) if tot_tp else "",
        "missed": "", "extra": "",
    })
    fieldnames = ["image", "tp", "fp", "fn", "precision", "recall", "f1",
                  "cat_correct", "cat_acc", "missed", "extra"]
    csv_path = write_eval_csv("fridge", rows, fieldnames)
    print(f"📝 결과 CSV 저장: {os.path.relpath(csv_path, _BASE_DIR)}")


if __name__ == "__main__":
    main()
