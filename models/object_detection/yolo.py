import os
from ultralytics import YOLO

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model = YOLO(os.path.join(_BASE_DIR, 'yolo_model', 'yolov8n.pt'))

results = model.predict(
    source=os.path.join(_BASE_DIR, 'picture_model', 'predict', 'fridge_test.jpeg'),
    save=True,
    device='mps'
)

# 3. 인식된 결과 JSON 형태로 구경해보기
for result in results:
    print(result.tojson())