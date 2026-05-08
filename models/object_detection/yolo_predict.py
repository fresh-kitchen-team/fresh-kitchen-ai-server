import os
from ultralytics import YOLO

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model = YOLO(os.path.join(_BASE_DIR, 'yolo_model', 'yolov8n.pt'))

results = model.predict(
    source=os.path.join(_BASE_DIR, 'picture_model', 'predict', 'fridge_test.jpeg'),
    conf=0.25,
    device='mps'
)

# 3. 결과 해석 및 출력
for result in results:
    boxes = result.boxes
    for box in boxes:
        # 핵심 수정:을 붙여서 값을 추출해야 에러가 안 납니다.
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf = float(box.conf)
        
        print(f"✅ 발견: [{cls_name}] - 확신도: {conf*100:.2f}%")

    # 4. 결과 이미지 저장
    result.save(filename=os.path.join(_BASE_DIR, 'yolo_model', 'yolo_result.jpg'))
    print(f"\n🖼️ 총 {len(boxes)}개의 물체를 찾았습니다. 결과는 'yolo_result.jpg' 확인!")