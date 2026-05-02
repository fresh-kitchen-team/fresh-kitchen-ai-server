import os
import random

# ==========================================
# 1. 설정 (종현님의 요구사항 반영)
# ==========================================
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(_BASE_DIR, 'dataset')
TRAIN_LIMIT = 400         # train 폴더의 목표 개수
VAL_LIMIT = 80         # val 폴더의 목표 개수

def limit_class_images(target_dir, limit):
    """지정된 폴더 내의 모든 클래스별 이미지 개수를 limit 이하로 맞춥니다."""
    
    if not os.path.exists(target_dir):
        print(f"⚠️ 경고: '{target_dir}' 폴더를 찾을 수 없습니다. 건너뜁니다.")
        return

    print(f"\n📂 '{target_dir}' 정리 시작 (목표: {limit}장 이하)")
    print("-" * 40)
    
    # 각 클래스 폴더(Apple, Banana 등)를 순회
    classes = sorted(os.listdir(target_dir))
    
    for class_name in classes:
        class_path = os.path.join(target_dir, class_name)
        
        # 폴더가 아니면(.DS_Store 등) 건너뜀
        if not os.path.isdir(class_path):
            continue

        # 이미지 파일만 리스트업 (대소문자 구분 없이)
        files = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        total_count = len(files)

        # 현재 개수가 목표치(limit)보다 많으면 삭제 진행
        if total_count > limit:
            delete_count = total_count - limit
            
            # 삭제할 파일 랜덤 선택 (중요: 랜덤이어야 데이터가 편향되지 않음)
            files_to_delete = random.sample(files, delete_count)
            
            for file in files_to_delete:
                file_path = os.path.join(class_path, file)
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"   ❌ 삭제 실패: {file} ({e})")
            
            print(f"   ✂️ [{class_name}] {total_count}장 -> {limit}장 ({delete_count}장 삭제)")
        else:
            print(f"   ✅ [{class_name}] {total_count}장 (유지 - 목표치 이내)")

def main():
    # 1. Train 폴더 정리 (500장 제한)
    train_dir = os.path.join(DATA_ROOT, 'train')
    limit_class_images(train_dir, TRAIN_LIMIT)

    # 2. Validation 폴더 정리 (100장 제한)
    val_dir = os.path.join(DATA_ROOT, 'val')
    limit_class_images(val_dir, VAL_LIMIT)

    print("\n✨ 모든 정리가 끝났습니다! 이제 학습 속도가 훨씬 빨라질 거예요.")

if __name__ == "__main__":
    main()