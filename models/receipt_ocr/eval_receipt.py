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

from models.eval_common import load_eval_samples, compare_items, write_eval_csv
from models.receipt_ocr.receipt_ocr import process_receipt_raw, filter_with_gemini

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(_BASE_DIR, "dataset", "eval", "receipt")
FIELDNAMES = ["사진", "구매일_예측", "구매일_정답", "예측", "정답", "놓침", "추가"]


def _names(items):
    return ", ".join((it.get("name") or "").strip() for it in items) or "(없음)"


def main():
    samples = load_eval_samples(EVAL_DIR)
    if not samples:
        print(f"❌ 평가할 샘플이 없습니다: {EVAL_DIR}")
        print("   이미지와 같은 이름의 정답 .json 을 넣어주세요. (예: receipt2.jpeg + receipt2.json)")
        return

    print(f"🧾 영수증 OCR 평가 — 샘플 {len(samples)}장\n")

    rows = []
    tot_match = tot_miss = tot_extra = 0
    date_correct = 0

    for img_path, gt in samples:
        name = os.path.basename(img_path)
        gt_items = gt.get("ingredients", [])
        raw = process_receipt_raw(img_path)
        result = filter_with_gemini(raw) if raw else {"purchasedAt": None, "ingredients": []}
        pred_items = result.get("ingredients", [])
        c = compare_items(pred_items, gt_items)

        pred_date = result.get("purchasedAt")
        gt_date = gt.get("purchasedAt")
        date_ok = (pred_date == gt_date)
        date_correct += int(date_ok)

        missed = [r["품목"] for r in c["rows"] if r["상태"] == "놓침"]
        extra = [r["품목"] for r in c["rows"] if r["상태"] == "추가"]
        rows.append({
            "사진": name,
            "구매일_예측": pred_date or "",
            "구매일_정답": gt_date or "",
            "예측": _names(pred_items),
            "정답": _names(gt_items),
            "놓침": ", ".join(missed),
            "추가": ", ".join(extra),
        })

        tot_match += c["match"]; tot_miss += c["miss"]; tot_extra += c["extra"]
        print(f"  {name:18s} 구매일 {'✅' if date_ok else '❌'}  일치 {c['match']:2d} / 놓침 {c['miss']:2d} / 추가 {c['extra']:2d}")

    csv_path = write_eval_csv("receipt", rows, FIELDNAMES)

    # ---- 맨 아래 합계 (구매일만 깔끔한 정답/오답, 품목은 거친 수치) ----
    date_acc = date_correct / len(samples) * 100
    print("\n" + "=" * 46)
    print(f"📊 구매일 정확도 {date_acc:5.1f}% ({date_correct}/{len(samples)})")
    print(f"   품목 합계  일치 {tot_match} / 놓침 {tot_miss} / 추가 {tot_extra}")
    print("   (놓침·추가엔 표기만 다른 같은 품목도 섞임 — 판단은 비교표로)")
    print("=" * 46)
    print(f"📝 비교 표 CSV: {os.path.relpath(csv_path, _BASE_DIR)}")


if __name__ == "__main__":
    main()
