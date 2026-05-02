import os
import random
import shutil
from pathlib import Path

# 1. 설정
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(_BASE_DIR, 'dataset', 'train')
VAL_DIR = os.path.join(_BASE_DIR, 'dataset', 'val')
SPLIT_RATIO = 0.2 # 20%를 이동
TARGET_CLASS = 'Mushroom'  # 👈 여기에 옮기고 싶은 폴더 이름을 적으세요!

def split_specific_class(class_name):
    train_class_path = Path(TRAIN_DIR) / class_name
    val_class_path = Path(VAL_DIR) / class_name

    # 1. 폴더 존재 확인
    if not train_class_path.exists():
        print(f"❌ '{TRAIN_DIR}' 안에 '{class_name}' 폴더가 없습니다. 이름을 확인해주세요.")
        return

    # 2. 검증용 폴더 생성
    val_class_path.mkdir(parents=True, exist_ok=True)

    # 3. 이미지 파일 목록 가져오기
    images = [f for f in train_class_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if len(images) == 0:
        print(f"⚠️ '{class_name}' 폴더에 이미지 파일이 없습니다.")
        return

    # 4. 랜덤 섞기 및 개수 계산
    random.shuffle(images)
    val_count = int(len(images) * SPLIT_RATIO)
    
    if val_count == 0:
        print(f"⚠️ 이미지가 너무 적어서(총 {len(images)}장) 나눌 수 없습니다.")
        return

    val_images = images[:val_count]

    # 5. 파일 이동
    print(f"🚀 '{class_name}' 데이터 이동 시작 (총 {len(images)}장 중 {val_count}장)")
    
    for img_path in val_images:
        shutil.move(str(img_path), str(val_class_path / img_path.name))

    print(f"✅ 완료: {class_name} 폴더에서 {val_count}장을 '{VAL_DIR}'로 옮겼습니다.")

if __name__ == "__main__":
    split_specific_class(TARGET_CLASS)