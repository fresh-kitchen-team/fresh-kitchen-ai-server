import os
import sys
import io
import time
import requests
from PIL import Image
from duckduckgo_search import DDGS

# 1. 환경 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# 2. 설정 변수
LIMIT = 200  # 쿼리당 목표 장수 (클래스당 최대 ~800장 → data_diet에서 400장으로 정리)
MIN_SIZE = 100       # 최소 이미지 크기 (px)
DOWNLOAD_TIMEOUT = 10
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(_BASE_DIR, 'dataset', 'crawldata')

# 3. 클래스별 크롤링 쿼리
# 키: 클래스 폴더명 / 값: 검색어 목록
CLASS_QUERIES = {

    '맥주': [
        '캔맥주 냉장고',
        '캔맥주 마트 진열',
        'canned beer refrigerator',
        '하이트 카스 테라 캔 맥주',
        '병맥주 냉장고',
        '병맥주 마트 진열',
        'beer bottle refrigerator',
        '하이트 카스 테라 병 맥주',
        '맥주 피쳐',
        '생맥주 피쳐 테이블',
        'beer pitcher',
        'draft beer pitcher',
    ],

    # ── 150장 미만 보강 클래스 ──────────────────────────────────────

    '무': [
        '무 마트 냉장고',
        '흰 무 채소 신선',
        '무 통째로 식재료',
        'daikon radish fresh',
    ],

    '라면사리': [
        '라면사리 봉지 마트',
        '라면사리 건면 식재료',
        '인스턴트 라면사리 포장',
        'ramen noodles package',
    ],

    '쌈장': [
        '쌈장 통 마트',
        '쌈장 용기 냉장고',
        '재래식 쌈장 식재료',
        'ssamjang korean paste',
    ],

    '식용유': [
        '식용유 병 마트',
        '식용유 냉장고 보관',
        '콩기름 포도씨유 식재료',
        'cooking oil bottle kitchen',
    ],

    '배': [
        '배 과일 마트',
        '한국 배 신선 냉장고',
        '배 과일 낱개',
        'korean pear fresh fruit',
    ],

    '옥수수': [
        '옥수수 냉장고',
        '옥수수 낱개 신선',
        '찐 옥수수 식재료',
        'corn cob fresh vegetable',
    ],

    '케첩': [
        '케첩 병 마트',
        '케첩 냉장고 보관',
        '토마토 케첩 용기',
        'ketchup bottle kitchen',
    ],

    '고등어': [
        '고등어 마트 생선',
        '고등어 냉장 신선',
        '고등어 한 마리 식재료',
        'mackerel fish fresh',
    ],

    '마요네즈': [
        '마요네즈 병 냉장고',
        '마요네즈 튜브 마트',
        '마요네즈 용기 식재료',
        'mayonnaise jar kitchen',
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


def download_image(url: str, save_path: str) -> bool:
    """URL에서 이미지를 다운로드하고 최소 크기를 확인합니다."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(resp.content)
        with Image.open(save_path) as img:
            w, h = img.size
            if min(w, h) < MIN_SIZE:
                os.remove(save_path)
                return False
        return True
    except Exception:
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def convert_to_jpg(folder_path):
    """폴더 내의 모든 이미지를 jpg로 변환하고 원본을 삭제합니다."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.jpg', '.jpeg')):
            continue
        try:
            with Image.open(file_path) as img:
                rgb_img = img.convert('RGB')
                new_file_path = os.path.splitext(file_path)[0] + "_conv.jpg"
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

    file_counter = count_images(class_dir)

    for i, query in enumerate(queries):
        tmp_dir = os.path.join(OUTPUT_ROOT, f'_tmp_{class_name}_{i}')
        os.makedirs(tmp_dir, exist_ok=True)

        print(f"   🔍 '{query}' ({limit}장 목표)")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(keywords=query, max_results=limit))

            tmp_counter = 0
            for result in results:
                url = result.get('image', '')
                if not url:
                    continue
                # 확장자 추출
                raw = url.split('?')[0].split('/')[-1]
                ext = '.jpg'
                if '.' in raw:
                    candidate = '.' + raw.rsplit('.', 1)[-1].lower()
                    if candidate in ('.jpg', '.jpeg', '.png', '.webp'):
                        ext = candidate
                tmp_counter += 1
                save_path = os.path.join(tmp_dir, f'{tmp_counter:06d}{ext}')
                download_image(url, save_path)

        except Exception as e:
            print(f"   ⚠️ DuckDuckGo 에러: {e}")

        # 임시 폴더 → 클래스 폴더로 이동 (파일명 중복 방지)
        for fname in sorted(os.listdir(tmp_dir)):
            src = os.path.join(tmp_dir, fname)
            if not os.path.isfile(src):
                continue
            ext = os.path.splitext(fname)[1].lower() or '.jpg'
            file_counter += 1
            dst = os.path.join(class_dir, f'{file_counter:06d}{ext}')
            os.rename(src, dst)

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
