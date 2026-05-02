import os
from PIL import Image, ImageFilter
from pathlib import Path

# 경로 설정 (종현님의 폴더 구조에 맞게 수정)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_VAL_DIR = os.path.join(_BASE_DIR, 'dataset', 'val')
DEST_BLUR_DIR = os.path.join(_BASE_DIR, 'dataset', 'val_blurred')
BLUR_RADIUS = 2  # 스마트폰 흔들림 정도 (1~5 사이 권장)

def make_blurred_dataset():
    source_path = Path(SOURCE_VAL_DIR)
    dest_path = Path(DEST_BLUR_DIR)

    if not source_path.exists():
        print(f"❌ '{SOURCE_VAL_DIR}' 폴더를 찾을 수 없습니다.")
        return

    print("📸 흐린 검증 데이터셋 생성을 시작합니다...")

    # 모든 하위 파일 탐색
    for file_path in source_path.rglob('*'):
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # 새 폴더 내의 상대 경로 계산
            relative_path = file_path.relative_to(source_path)
            new_file_path = dest_path / relative_path
            
            # 하위 폴더(Pork, Salt 등)가 없으면 생성
            new_file_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with Image.open(file_path) as img:
                    # 가우시안 블러 적용 및 저장
                    # RGB 모드로 변환해야 JPG 저장 시 에러가 안 납니다
                    blurred_img = img.convert('RGB').filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
                    blurred_img.save(new_file_path)
            except Exception as e:
                print(f"⚠️ {file_path.name} 처리 중 에러: {e}")

    print(f"✅ 완료! '{DEST_BLUR_DIR}' 폴더에 모든 클래스 구조가 복사되었습니다.")

if __name__ == "__main__":
    make_blurred_dataset()