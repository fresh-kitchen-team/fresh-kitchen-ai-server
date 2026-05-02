import os
import time
import sys
import io
import ssl
from bing_image_downloader import downloader
from PIL import Image  # 이미지 변환을 위한 라이브러리

# 1. 환경 설정
ssl._create_default_https_context = ssl._create_unverified_context
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# 2. 설정 변수
LIMIT = 150
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(_BASE_DIR, 'dataset', 'crawldata')

queries = [
    "새송이버섯 1봉지",
    "팽이버섯 묶음",
    "느타리버섯 팩",
    "표고버섯 팩",
    "양송이버섯 봉지"
]
def convert_to_jpg(folder_path):
    """폴더 내의 모든 이미지를 jpg로 변환하고 원본을 삭제합니다."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 파일이 이미 jpg인 경우 패스 (대문자 JPG 포함)
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            continue
            
        try:
            with Image.open(file_path) as img:
                # RGBA(PNG 등)를 RGB로 변환 (JPG는 투명도를 지원하지 않음)
                rgb_img = img.convert('RGB')
                # 확장자를 .jpg로 바꿔서 저장
                new_file_path = os.path.splitext(file_path)[0] + ".jpg"
                rgb_img.save(new_file_path, "JPEG")
            
            # 원본 파일(png, webp 등) 삭제
            os.remove(file_path)
        except Exception as e:
            # 이미지가 아니거나 변환 불가능한 파일 삭제
            if os.path.isfile(file_path):
                os.remove(file_path)

def run_project_crawler():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    for q in queries:
        try:
            print(f"\n🔍 '{q}' 수집 및 JPG 변환 중...")
            
            # 1. 일단 다운로드
            downloader.download(q, limit=LIMIT, output_dir=OUTPUT_ROOT, 
                                adult_filter_off=True, force_replace=False, 
                                timeout=10, verbose=False)
            
            # 2. 다운로드된 폴더 경로 파악
            target_folder = os.path.join(OUTPUT_ROOT, q)
            
            # 3. JPG로 일괄 변환 및 정리
            if os.path.exists(target_folder):
                convert_to_jpg(target_folder)
                print(f"✨ '{q}' 폴더 정리 완료 (Only JPG)")
            
            time.sleep(1)

        except Exception as e:
            print(f"⚠️ 에러 발생: {e}")
            continue

    print("\n" + "="*50 + "\n✅ 모든 파일이 .jpg로 저장되었습니다.\n" + "="*50)

if __name__ == "__main__":
    run_project_crawler()



