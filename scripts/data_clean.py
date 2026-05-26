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

def clean_entire_dataset(root_dir):
    root_path = Path(root_dir)
    hashes = {}
    total_deleted = 0

    print(f"🚀 전체 데이터셋({root_dir}) 정리를 시작합니다...")

    # rglob('*')을 쓰면 하위 폴더(Apple, Banana 등)를 모두 자동으로 탐색합니다.
    for file_path in root_path.rglob('*'):
        # 이미지 파일만 골라내기
        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            file_hash = get_file_hash(file_path)

            if file_hash in hashes:
                # 중복 발견!
                print(f"🗑️ 중복 삭제: {file_path.relative_to(root_path)} (원본: {hashes[file_hash]})")
                os.remove(file_path)
                total_deleted += 1
            else:
                # 새로운 이미지면 해시 등록 (경로 저장)
                hashes[file_hash] = file_path.relative_to(root_path)

    print(f"\n✅ 완료! 총 {total_deleted}개의 중복 사진을 삭제했습니다.")

if __name__ == "__main__":
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_DIR = os.path.join(_BASE_DIR, 'dataset', 'train')
    clean_entire_dataset(BASE_DIR)