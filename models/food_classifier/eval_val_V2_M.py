import os
import csv
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. 설정
# ==========================================
# 별도 held-out test set 이 없어 검증(val) 셋 기준으로 진단한다.
# val 은 early stopping·모델 선택에 사용된 셋이라 절대 수치는 다소 낙관적일 수 있으나,
# "어떤 클래스가 무엇과 혼동되는가" 라는 혼동 패턴 진단에는 충분하다.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_BASE_DIR, 'dataset', 'val')
MODEL_PATH = os.path.join(_BASE_DIR, 'best_food_model_v2_m_ver5.pth')
CM_CSV_PATH = os.path.join(_BASE_DIR, 'docs', 'logs', 'confusion_matrix_val.csv')
BATCH_SIZE = 8   # MPS 안정성 최우선
NUM_WORKERS = 0  # MPS는 멀티프로세싱 비활성화 (메모리 안정성)
TOP_CONFUSION = 15  # 출력할 최다 혼동 쌍 개수


def main():
    # --------------------------------------
    # 2. 장치 설정
    # --------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Mac GPU(MPS)로 평가를 진행합니다.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 NVIDIA GPU로 평가를 진행합니다.")
    else:
        device = torch.device("cpu")
        print("⚠️ CPU로 평가합니다.")

    # --------------------------------------
    # 3. 검증(val) 데이터 불러오기
    # --------------------------------------
    if not os.path.exists(DATA_DIR):
        print(f"❌ 오류: '{DATA_DIR}' 폴더가 없습니다. 경로를 확인해주세요.")
        return

    # 학습/추론과 동일한 전처리
    eval_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_dataset = datasets.ImageFolder(DATA_DIR, eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    # --------------------------------------
    # 4. 모델 불러오기
    # --------------------------------------
    print("\n🤖 모델을 불러오는 중...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        class_names = checkpoint['class_names']
        print(f"\n📂 총 {len(class_names)}개 클래스 / 검증 이미지 {len(eval_dataset)}장")

        # 폴더 클래스 순서가 학습 시와 다르면 라벨 인덱스가 어긋나 혼동행렬이 틀어짐
        if eval_dataset.classes != class_names:
            print("❌ 오류: 검증 폴더의 클래스 순서가 모델과 다릅니다.")
            print(f"   모델: {class_names}")
            print(f"   검증: {eval_dataset.classes}")
            return
    except FileNotFoundError:
        print(f"❌ 오류: '{MODEL_PATH}' 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        return

    model = models.efficientnet_v2_m(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(num_ftrs, len(class_names))
    )

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ EfficientNet V2-M 모델을 성공적으로 불러왔습니다!")
    except Exception as e:
        print(f"❌ 가중치 로드 중 오류 발생: {e}")
        return

    model = model.to(device)
    model.eval()

    # --------------------------------------
    # 5. 추론 + 혼동행렬 누적
    # --------------------------------------
    print("\n🔥 평가 시작! (잠시만 기다려주세요...)")
    n = len(class_names)
    # confusion[true][pred] = 개수
    confusion = torch.zeros((n, n), dtype=torch.long)

    with torch.no_grad():
        num_batches = len(eval_loader)
        for batch_idx, (inputs, labels) in enumerate(eval_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu()

            for t, p in zip(labels, preds):
                confusion[t.item()][p.item()] += 1

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                print(f"  진행 중: Batch {batch_idx + 1}/{num_batches}", end='\r', flush=True)

    # --------------------------------------
    # 6. 지표 계산
    # --------------------------------------
    diag = confusion.diag()
    class_total = confusion.sum(dim=1)            # 클래스별 실제 개수
    total_correct = int(diag.sum().item())
    total = int(confusion.sum().item())
    overall_acc = 100.0 * total_correct / total if total else 0.0

    print("\n" + "=" * 50)
    print(f"🏆 검증(val) 정확도: {overall_acc:.2f}% ({total_correct}/{total})")
    print("   ※ held-out test set 이 아닌 val 기준 — 절대 수치는 다소 낙관적일 수 있음")
    print("=" * 50)

    # 6-1. 클래스별 정확도 (낮은 순)
    print("\n📊 [클래스별 정확도 — 낮은 순]")
    per_class = []
    for i in range(n):
        tot = int(class_total[i].item())
        acc = 100.0 * int(diag[i].item()) / tot if tot > 0 else 0.0
        per_class.append((acc, i, tot))
    for acc, i, tot in sorted(per_class, key=lambda x: x[0]):
        if tot > 0:
            bar = "🟩" * int(acc // 20) + "⬜" * (5 - int(acc // 20))
            print(f"   {class_names[i]:<15} : {acc:>6.1f}% {bar} ({int(diag[i].item())}/{tot})")
        else:
            print(f"   {class_names[i]:<15} : 데이터 없음")

    # 6-2. 최다 혼동 쌍 (off-diagonal Top N)
    print(f"\n🔀 [최다 혼동 쌍 Top {TOP_CONFUSION}] (실제 → 예측 : 횟수)")
    pairs = []
    for t in range(n):
        for p in range(n):
            if t != p and confusion[t][p] > 0:
                pairs.append((int(confusion[t][p].item()), t, p))
    pairs.sort(reverse=True)
    if not pairs:
        print("   혼동 없음 (완벽 분류)")
    for cnt, t, p in pairs[:TOP_CONFUSION]:
        tot = int(class_total[t].item())
        ratio = 100.0 * cnt / tot if tot else 0.0
        print(f"   {class_names[t]:<10} → {class_names[p]:<10} : {cnt:>3}회 "
              f"(해당 클래스의 {ratio:.1f}%)")

    # --------------------------------------
    # 7. 전체 혼동행렬 CSV 저장 (발표/기록용)
    # --------------------------------------
    os.makedirs(os.path.dirname(CM_CSV_PATH), exist_ok=True)
    with open(CM_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # 헤더: 좌상단 라벨 + 예측 클래스명
        writer.writerow(["true\\pred"] + class_names)
        for i in range(n):
            writer.writerow([class_names[i]] + [int(confusion[i][j].item()) for j in range(n)])
    print(f"\n💾 전체 혼동행렬 저장 → {os.path.relpath(CM_CSV_PATH, _BASE_DIR)}")
    print("\n✅ 평가 완료!")


if __name__ == '__main__':
    main()
