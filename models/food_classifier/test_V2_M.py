import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# ==========================================
# 1. 설정
# ==========================================
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_BASE_DIR, 'dataset', 'test')
MODEL_PATH = os.path.join(_BASE_DIR, 'best_food_model_v2_m_ver4.pth')
BATCH_SIZE = 8  # MPS 안정성 최우선
NUM_WORKERS = 0  # MPS는 멀티프로세싱 비활성화 (메모리 안정성)

def main():
    # --------------------------------------
    # 2. 장치 설정 (MacBook MPS 기준)
    # --------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"🚀 Mac GPU(MPS)로 테스트를 진행합니다.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 NVIDIA GPU로 테스트를 진행합니다.")
    else:
        device = torch.device("cpu")
        print("⚠️ CPU로 테스트합니다.")

    # --------------------------------------
    # 3. 테스트 데이터 불러오기
    # --------------------------------------
    if not os.path.exists(DATA_DIR):
        print(f"❌ 오류: '{DATA_DIR}' 폴더가 없습니다. 경로를 확인해주세요.")
        return

    # V2-M 학습 때와 동일한 전처리
    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(DATA_DIR, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)

    # --------------------------------------
    # 4. 모델 불러오기
    # --------------------------------------
    print("\n🤖 모델을 불러오는 중...")

    # 1) 체크포인트 먼저 로드해서 class_names 확보 (폴더 구조 의존 제거)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        class_names = checkpoint['class_names']
        print(f"\n📂 총 {len(class_names)}개 클래스의 테스트 이미지를 찾았습니다.")
        print(f"   - 전체 테스트 이미지 수: {len(test_dataset)}장")

        # 테스트 폴더 클래스 순서가 학습 시와 다르면 정확도가 잘못 측정됨
        if test_dataset.classes != class_names:
            print("❌ 오류: 테스트 폴더의 클래스 순서가 모델과 다릅니다.")
            print(f"   모델: {class_names}")
            print(f"   테스트: {test_dataset.classes}")
            return
    except FileNotFoundError:
        print(f"❌ 오류: '{MODEL_PATH}' 파일을 찾을 수 없습니다.")
        print("   💡 먼저 train_EfficientNet_V2_M.py를 실행해서 모델을 학습해주세요.")
        return
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        return

    # 2) 모델 구조 생성 (EfficientNet_V2-M)
    model = models.efficientnet_v2_m(weights=None)

    # 3) 마지막 레이어 수정 (체크포인트 클래스 수 기준)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(num_ftrs, len(class_names))
    )

    # 4) 가중치 적재
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ EfficientNet V2-M 모델을 성공적으로 불러왔습니다!")
    except Exception as e:
        print(f"❌ 가중치 로드 중 오류 발생: {e}")
        print("💡 팁: 학습할 때 클래스 개수와 테스트 폴더의 클래스 개수가 같은지 확인하세요.")
        return

    model = model.to(device)
    model.eval() # 평가 모드

    # --------------------------------------
    # 5. 테스트
    # --------------------------------------
    print("\n🔥 채점 시작! (잠시만 기다려주세요...)")
    
    running_corrects = 0
    
    # 클래스별 정답률 계산을 위한 변수들
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))

    with torch.no_grad(): # 테스트 - 기울기 계산 X
        num_batches = len(test_loader)
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 전체 정답 수
            running_corrects += torch.sum(preds == labels.data)

            # 클래스별 정답 수
            c = (preds == labels).reshape(-1)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            # 진행 상황 표시
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                print(f"  테스트 진행 중: Batch {batch_idx + 1}/{num_batches}", end='\r', flush=True)

    # --------------------------------------
    # 6. 최종 결과 출력
    # --------------------------------------
    total_acc = running_corrects.float() / len(test_dataset)

    print("\n" + "="*50)
    print(f"🏆 최종 테스트 정확도: {total_acc * 100:.2f}% ({int(running_corrects)}/{len(test_dataset)})")
    print("="*50)
    print("\n📊 [음식 클래스별 상세 정확도]")

    sorted_indices = sorted(range(len(class_names)),
                           key=lambda i: class_correct[i]/class_total[i] if class_total[i] > 0 else 0)

    for i in sorted_indices:
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            bar = "🟩" * int(acc // 20) + "⬜" * (5 - int(acc // 20))
            print(f"   {class_names[i]:<15} : {acc:>6.1f}% {bar} ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f"   {class_names[i]:<15} : 데이터 없음")

    print("\n✅ 테스트 완료!")

if __name__ == '__main__':
    main()