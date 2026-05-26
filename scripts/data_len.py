import os
from pathlib import Path

def count_dataset_images(root_dir):
    root_path = Path(root_dir)
    # 우리가 체크할 메인 폴더들
    splits = ['train', 'val', 'test']
    
    print(f"📊 데이터셋 현황 보고서 ({root_dir})")
    print("=" * 40)

    for split in splits:
        split_path = root_path / split
        if not split_path.exists():
            print(f"⚠️ '{split}' 폴더를 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        print(f"\n[📂 {split.upper()} 폴더]")
        # 하위 폴더(클래스) 목록 가져오기 및 정렬
        class_folders = sorted([f for f in split_path.iterdir() if f.is_dir()])
        
        total_images = 0
        for class_folder in class_folders:
            # 이미지 확장자들만 카운트
            images = [f for f in class_folder.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            count = len(images)
            total_images += count
            print(f" - {class_folder.name:15}: {count}장")
        print(f"📈 {split} 전체 합계: {total_images}장")
    print("=" * 40)

if __name__ == "__main__":
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_DIR = os.path.join(_BASE_DIR, 'dataset')
    count_dataset_images(BASE_DIR)