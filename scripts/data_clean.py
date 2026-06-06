import os
import hashlib
from pathlib import Path

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(8192)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(8192)
    return hasher.hexdigest()

def clean_dataset(target_dirs):
    """여러 폴더를 하나의 해시 맵으로 함께 검사 — split 간 데이터 누수까지 차단한다.

    auto_labeled / test_real_image 등은 호출 측에서 제외하므로, 자동 라벨 결과나
    실제 테스트 이미지가 중복으로 오인되어 삭제되는 일이 없다.
    """
    hashes = {}
    total_deleted = 0

    for root_dir in target_dirs:
        root_path = Path(root_dir)
        if not root_path.exists():
            print(f"⚠️ '{root_path.name}' 폴더 없음, 건너뜀")
            continue

        print(f"🚀 '{root_path.name}' 정리 중...")
        # rglob('*')을 쓰면 하위 폴더(클래스별 폴더)를 모두 자동으로 탐색합니다.
        for file_path in root_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                file_hash = get_file_hash(file_path)
                if file_hash in hashes:
                    print(f"🗑️ 중복 삭제: {file_path} (원본: {hashes[file_hash]})")
                    os.remove(file_path)
                    total_deleted += 1
                else:
                    hashes[file_hash] = file_path

    print(f"\n✅ 완료! 총 {total_deleted}개의 중복 사진을 삭제했습니다.")

if __name__ == "__main__":
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(_BASE_DIR, 'dataset')
    # 중복 제거 대상: 크롤링 원본 + 학습 split (split 간 데이터 누수도 함께 차단)
    # auto_labeled / test_real_image 는 보존해야 하므로 제외
    TARGET_DIRS = [os.path.join(DATASET_DIR, d) for d in ('crawldata', 'train', 'val', 'test')]
    clean_dataset(TARGET_DIRS)