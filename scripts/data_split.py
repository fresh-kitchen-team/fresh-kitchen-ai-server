import os
import random
import shutil
from pathlib import Path

# 1. 설정
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(_BASE_DIR, 'dataset', 'train')
VAL_DIR = os.path.join(_BASE_DIR, 'dataset', 'val')
SPLIT_RATIO = 0.2  # train에서 val로 이동할 비율 (20%)

def split_all_classes():
    train_path = Path(TRAIN_DIR)

    if not train_path.exists():
        print(f"❌ '{TRAIN_DIR}' 폴더가 없습니다.")
        return

    classes = sorted([f for f in train_path.iterdir() if f.is_dir()])
    print(f"총 {len(classes)}개 클래스 처리 시작 (val 비율: {SPLIT_RATIO*100:.0f}%)")
    print("-" * 50)

    total_moved = 0
    for class_dir in classes:
        class_name = class_dir.name
        val_class_path = Path(VAL_DIR) / class_name
        # 멱등성 보장: 이미 분리된 val 이미지가 있으면 초기화
        if val_class_path.exists():
            shutil.rmtree(val_class_path)
        val_class_path.mkdir(parents=True, exist_ok=True)

        images = [f for f in class_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        if len(images) == 0:
            print(f"⚠️  [{class_name}] 이미지 없음, 건너뜀")
            continue

        random.shuffle(images)
        val_count = int(len(images) * SPLIT_RATIO)

        if val_count == 0:
            print(f"⚠️  [{class_name}] 이미지 부족({len(images)}장), 건너뜀")
            continue

        for img_path in images[:val_count]:
            shutil.move(str(img_path), str(val_class_path / img_path.name))

        total_moved += val_count
        print(f"✅ [{class_name:<15}] {len(images)}장 → val {val_count}장 이동")

    print(f"\n🎉 완료! 총 {total_moved}장을 val로 이동했습니다.")

if __name__ == "__main__":
    split_all_classes()
