"""영수증 OCR 평가 — 정답 vs 예측을 품목별 표(CSV)로 비교.

dataset/eval/receipt/ 안의 영수증 사진마다 같은 이름의 정답 JSON을 두고,
process_receipt_raw() + filter_with_gemini() 예측을 정답과 품목별로 비교해 CSV 표로 저장한다.
CSV 열: 사진 | 품목 | 상태(일치/놓침/추가) | 정답_카테고리 | 예측_카테고리
구매일(purchasedAt)은 사진별로 한 행(품목="📅 구매일")으로 함께 기록.

정답 JSON 형식 (예: receipt2.jpeg → receipt2.json):
    {"purchasedAt": "2026-05-13",
     "ingredients": [{"name": "두부", "category": "GRAIN"}, ...]}
    구매일이 영수증에 없으면 "purchasedAt": null.

실행:
    python -m models.receipt_ocr.eval_receipt
"""

import os

from models.eval_common import load_eval_samples, compare_items, prf, write_eval_csv
from models.receipt_ocr.receipt_ocr import process_receipt_raw, filter_with_gemini

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(_BASE_DIR, "dataset", "eval", "receipt")
FIELDNAMES = ["사진", "품목", "상태", "정답_카테고리", "예측_카테고리"]


def main():
    samples = load_eval_samples(EVAL_DIR)
    if not samples:
        print(f"❌ 평가할 샘플이 없습니다: {EVAL_DIR}")
        print("   이미지와 같은 이름의 정답 .json 을 넣어주세요. (예: receipt2.jpeg + receipt2.json)")
        return

    print(f"🧾 영수증 OCR 평가 — 샘플 {len(samples)}장\n")

    rows = []
    tot_match = tot_miss = tot_extra = tot_cat = 0
    date_correct = 0

    for img_path, gt in samples:
        name = os.path.basename(img_path)
        raw = process_receipt_raw(img_path)
        result = filter_with_gemini(raw) if raw else {"purchasedAt": None, "ingredients": []}
        c = compare_items(result.get("ingredients", []), gt.get("ingredients", []))

        # 구매일 비교 행 (사진마다 1줄)
        pred_date = result.get("purchasedAt")
        gt_date = gt.get("purchasedAt")
        date_ok = (pred_date == gt_date)
        date_correct += int(date_ok)
        rows.append({"사진": name, "품목": "📅 구매일",
                     "상태": "일치" if date_ok else "불일치",
                     "정답_카테고리": gt_date or "", "예측_카테고리": pred_date or ""})

        for r in c["rows"]:
            rows.append({"사진": name, **r})

        tot_match += c["match"]; tot_miss += c["miss"]; tot_extra += c["extra"]; tot_cat += c["cat_correct"]
        print(f"  {name:18s} 구매일 {'✅' if date_ok else '❌'}  일치 {c['match']:2d} / 놓침 {c['miss']:2d} / 추가 {c['extra']:2d}")

    csv_path = write_eval_csv("receipt", rows, FIELDNAMES)

    # ---- 맨 아래 전체 수치 ----
    P, R, F1 = prf(tot_match, tot_extra, tot_miss)
    cat_acc = (tot_cat / tot_match * 100) if tot_match else 0.0
    date_acc = date_correct / len(samples) * 100

    print("\n" + "=" * 46)
    print("📊 전체 수치")
    print(f"   구매일 정확도 {date_acc:5.1f}% ({date_correct}/{len(samples)})")
    print(f"   품목 일치 {tot_match} / 놓침 {tot_miss} / 추가 {tot_extra}")
    print(f"   Precision {P*100:5.1f}%   Recall {R*100:5.1f}%   F1 {F1*100:5.1f}%")
    print(f"   카테고리 일치(일치 품목 중) {cat_acc:5.1f}% ({tot_cat}/{tot_match})")
    print("=" * 46)
    print(f"📝 비교 표 CSV: {os.path.relpath(csv_path, _BASE_DIR)}")


if __name__ == "__main__":
    main()
