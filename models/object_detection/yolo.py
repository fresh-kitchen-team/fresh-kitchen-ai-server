from ultralytics import YOLO

# 1. 생짜 모델 불러오기 (n은 nano 버전으로 가장 가볍고 빠릅니다)
# 처음 실행하면 약 6MB 정도 되는 가중치 파일을 자동으로 다운로드합니다.
model = YOLO('yolov8n.pt') 

# 2. 냉장고 사진 테스트 (파일 경로를 종현 님 사진으로 바꿔주세요)
# device='mps'를 넣으면 맥북 GPU를 사용합니다.
results = model.predict(source='predict/fridge_test.jpg', save=True, device='mps')

# 3. 인식된 결과 JSON 형태로 구경해보기
for result in results:
    print(result.tojson())