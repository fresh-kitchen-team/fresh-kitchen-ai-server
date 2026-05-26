import os
import sys
import io
import time
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from PIL import Image

# 1. 환경 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# 2. 설정 변수
LIMIT = 200  # 쿼리당 목표 장수 (Google+Bing 각각, 클래스당 최대 ~800장 → data_diet에서 400장으로 정리)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(_BASE_DIR, 'dataset', 'crawldata')

# 3. 클래스별 크롤링 쿼리
# 키: 클래스 폴더명 / 값: 검색어 목록
CLASS_QUERIES = {
    '무': [
        '무 마트 진열 채소',
        '흰 무 낱개 신선',
        '무 냉장 채소 코너',
        'daikon radish supermarket',
    ],

    '라면': [
        '라면 봉지 마트 진열',
        '신라면 진라면 봉지 제품',
        '인스턴트 라면 상품 사진',
        'instant ramen noodles package product',
    ],

    '쌈장': [
        '쌈장 용기 마트 진열',
        '쌈장 통 제품 사진',
        '청정원 재래식 쌈장 상품',
        'ssamjang korean paste container product',
    ],

    '식용유': [
        '식용유 병 마트 진열',
        '해표 백설 식용유 제품',
        '식용유 상품 사진',
        'cooking oil bottle supermarket product',
    ],

    '배': [
        '배 과일 마트 진열',
        '신고배 한국 배 낱개',
        '배 과일 냉장 신선',
        'korean pear fruit supermarket',
    ],

    '옥수수': [
        '옥수수 마트 진열 채소',
        '신선 옥수수 낱개 채소',
        '옥수수 냉장 채소 코너',
        'corn cob supermarket fresh vegetable',
    ],

    '케첩': [
        '케첩 병 마트 진열',
        '하인즈 오뚜기 케첩 제품',
        '토마토케첩 상품 사진',
        'ketchup bottle product supermarket',
    ],

    '고등어': [
        '고등어 마트 생선 코너',
        '고등어 냉장 신선 생선',
        '고등어 낱개 생선 가게',
        'mackerel fish supermarket fresh',
    ],

    '마요네즈': [
        '마요네즈 병 마트 진열',
        '오뚜기 마요네즈 제품 사진',
        '마요네즈 튜브 상품',
        'mayonnaise jar product supermarket',
    ],

    '토마토': [
        '토마토 냉장고 신선',
        '토마토 낱개 마트',
        '방울토마토 식재료',
        'tomato fresh vegetable',
    ],

    '깻잎': [
        '깻잎 냉장고',
        '깻잎 묶음 마트',
        '신선 깻잎 채소',
        'perilla leaf korean vegetable',
    ],

    '베이컨': [
        '베이컨 냉장고 포장',
        '베이컨 슬라이스 마트',
        '베이컨 팩 식재료',
        'bacon package refrigerator',
    ],

    '콜라': [
        '콜라 캔 냉장고',
        '콜라 병 마트',
        '코카콜라 펩시 냉장고',
        'cola can refrigerator',
    ],

    '버터': [
        '버터 냉장고 보관',
        '버터 포장 마트',
        '무염 버터 식재료',
        'butter block refrigerator',
    ],

    '설탕': [
        '설탕 봉지 마트',
        '설탕 통 식재료',
        '흰 설탕 용기',
        'sugar bag kitchen',
    ],

    '부추': [
        '부추 냉장고',
        '부추 묶음 마트',
        '신선 부추 채소',
        'chives korean vegetable fresh',
    ],

    '당면': [
        '당면 봉지 마트',
        '당면 식재료 건면',
        '국수 당면 포장',
        'glass noodles cellophane noodles package',
    ],
}


def convert_to_jpg(folder_path):
    """폴더 내의 모든 이미지를 jpg로 변환하고 원본을 삭제합니다."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            continue
        try:
            with Image.open(file_path) as img:
                rgb_img = img.convert('RGB')
                new_file_path = os.path.splitext(file_path)[0] + "_conv.jpg"
                # 변환 파일명 충돌 방지
                if os.path.exists(new_file_path):
                    new_file_path = os.path.splitext(file_path)[0] + f"_{int(time.time())}.jpg"
                rgb_img.save(new_file_path, "JPEG")
            os.remove(file_path)
        except Exception:
            if os.path.isfile(file_path):
                os.remove(file_path)


def count_images(folder_path):
    return len([f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


def crawl_class(class_name, queries, limit):
    """쿼리당 limit장씩 임시 서브폴더에 저장 후 클래스 폴더로 합칩니다."""
    class_dir = os.path.join(OUTPUT_ROOT, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # 현재 저장된 파일 수 기준으로 번호 시작점 결정
    file_counter = count_images(class_dir)

    for i, query in enumerate(queries):
        tmp_dir = os.path.join(OUTPUT_ROOT, f'_tmp_{class_name}_{i}')
        os.makedirs(tmp_dir, exist_ok=True)

        print(f"   🔍 '{query}' ({limit}장 목표)")

        # 1차: Google
        try:
            google_crawler = GoogleImageCrawler(
                downloader_threads=2,
                storage={'root_dir': tmp_dir},
                log_level=40
            )
            google_crawler.crawl(keyword=query, max_num=limit, min_size=(100, 100))
        except Exception as e:
            print(f"   ⚠️ Google 에러: {e}")

        # 2차: Bing
        try:
            bing_crawler = BingImageCrawler(
                downloader_threads=2,
                storage={'root_dir': tmp_dir},
                log_level=40
            )
            bing_crawler.crawl(keyword=query, max_num=limit, min_size=(100, 100))
        except Exception as e:
            print(f"   ⚠️ Bing 에러: {e}")

        # 임시 폴더 → 클래스 폴더로 이동 (파일명 중복 방지)
        for fname in os.listdir(tmp_dir):
            src = os.path.join(tmp_dir, fname)
            if not os.path.isfile(src):
                continue
            ext = os.path.splitext(fname)[1].lower() or '.jpg'
            file_counter += 1
            dst = os.path.join(class_dir, f'{file_counter:06d}{ext}')
            os.rename(src, dst)

        # 임시 폴더 삭제
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass

        time.sleep(2)

    convert_to_jpg(class_dir)
    final = count_images(class_dir)
    print(f"   ✅ {class_name} 완료 — 총 {final}장")


def run_project_crawler():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for class_name, queries in CLASS_QUERIES.items():
        print(f"\n📦 [{class_name}] 수집 시작")
        try:
            crawl_class(class_name, queries, limit=LIMIT)
        except Exception as e:
            print(f"⚠️ [{class_name}] 에러: {e}")
            continue

    print("\n" + "="*50 + "\n✅ 전체 크롤링 완료\n" + "="*50)


if __name__ == "__main__":
    run_project_crawler()
