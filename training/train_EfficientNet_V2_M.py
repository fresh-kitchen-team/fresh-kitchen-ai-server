import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import time
import copy
import csv
from PIL import Image

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE_DIR, 'dataset')
BATCH_SIZE = 8  # MPS 안정성 최우선
LEARNING_RATE = 1e-4
EPOCHS = 30
PATIENCE = 5
SAVE_PATH = os.path.join(_BASE_DIR, 'best_food_model_v2_m_ver2.pth')
LOG_PATH = os.path.join(_BASE_DIR, 'training_log.csv')
NUM_WORKERS = 0 # MPS는 멀티프로세싱 비활성화 (메모리 안정성)

def remove_corrupted_images(directory):
    print("🔍 손상된 이미지 검사를 시작합니다...")
    removed_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.load()
                except (IOError, SyntaxError):
                    print(f"   ⚠️ 손상된 이미지 삭제됨: {file_path}")
                    os.remove(file_path)
                    removed_count += 1
    print(f"✅ 검사 완료! 총 {removed_count}개의 손상된 이미지가 제거되었습니다.\n")

def main():
    # --------------------------------------
    # 2. 하드웨어 가속 설정
    # --------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Mac GPU(MPS) 가속을 사용합니다!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 NVIDIA GPU(CUDA)를 사용합니다.")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU가 없어 CPU를 사용합니다.")

    # --------------------------------------
    # 3. 데이터 사전 검사
    # --------------------------------------
    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print(f"❌ 에러: {DATA_DIR}/train 폴더를 찾을 수 없습니다.")
        return

    remove_corrupted_images(DATA_DIR)

    # --------------------------------------
    # 4. 데이터 전처리 (스마트폰 촬영 환경 최적화)
    # --------------------------------------
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(480, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(480),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --------------------------------------
    # 5. 데이터 로딩
    # --------------------------------------
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # NUM_WORKERS=0이므로 pin_memory 비활성화 (메모리 절약)
    pin_memory = False

    # 클래스 불균형 해결: WeightedRandomSampler
    # 적은 클래스가 더 자주 뽑히도록 가중치 부여
    train_targets = image_datasets['train'].targets
    class_sample_counts = [train_targets.count(i) for i in range(len(class_names))]
    class_weights = [1.0 / count if count > 0 else 0.0 for count in class_sample_counts]
    sample_weights = [class_weights[t] for t in train_targets]

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=train_sampler,
                            num_workers=NUM_WORKERS),
        'val':   DataLoader(image_datasets['val'],   batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    }

    print(f"✅ 학습 클래스 ({len(class_names)}개): {class_names}")
    print(f"   - Train: {dataset_sizes['train']}장 / Val: {dataset_sizes['val']}장")

    # 클래스 불균형 확인
    print("\n📊 클래스별 Train 이미지 수:")
    for cls in class_names:
        count = len(os.listdir(os.path.join(DATA_DIR, 'train', cls)))
        print(f"   {cls:<15}: {count}장")

    # --------------------------------------
    # 6. 모델 설계 (EfficientNet-V2-M)
    # --------------------------------------
    print("\n🤖 EfficientNet_V2-M 모델 로드 중...")
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[-4:].parameters():
        param.requires_grad = True

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(num_ftrs, len(class_names))
    )
    model = model.to(device)

    # --------------------------------------
    # 7. 손실 함수 / Optimizer / Scheduler
    # --------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=5e-3
    )
    # ReduceLROnPlateau: val_loss가 개선 없으면 lr 감소 (조기 종료와 완벽 호환)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # --------------------------------------
    # 8. 학습 루프
    # --------------------------------------
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss_for_log = float('inf')
    patience_check = 0

    # CSV 로그 초기화
    with open(LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    print(f"\n🔥 학습 시작!")
    print("-" * 30)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        epoch_log = {'epoch': epoch + 1}

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0
            num_batches = len(dataloaders[phase])

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                print(f'    [{phase.upper()}] Batch {batch_idx + 1}/{num_batches} - 데이터 로딩 중...', end='\r', flush=True)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 진행 상황 출력 (매 배치마다)
                print(f'    [{phase.upper()}] Batch {batch_idx + 1}/{num_batches}', end='\r', flush=True)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'\n  {phase.upper()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

            epoch_log[f'{phase}_loss'] = round(epoch_loss, 4)
            epoch_log[f'{phase}_acc'] = round(epoch_acc.item(), 4)

            if phase == 'val':
                current_lr = optimizer.param_groups[0]['lr']
                epoch_log['lr'] = current_lr

                # Scheduler는 val_loss 기준으로 lr 조정
                scheduler.step(epoch_loss)

                # Best 모델 기준: val_acc
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss_for_log = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_check = 0
                    print(f"   🎉 최고 정확도 갱신! (Acc: {best_acc*100:.2f}%)")
                else:
                    patience_check += 1
                    print(f"   ⚠️ 성능 향상 없음 ({patience_check}/{PATIENCE})")

        # CSV에 epoch 결과 기록
        with open(LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch_log['epoch'],
                epoch_log['train_loss'], epoch_log['train_acc'],
                epoch_log['val_loss'],   epoch_log['val_acc'],
                epoch_log['lr']
            ])

        if patience_check >= PATIENCE:
            print(f"⏹️  조기 종료! (Val Acc {PATIENCE}번 연속 향상 없음)")
            break
        print("-" * 30)

    # --------------------------------------
    # 9. 클래스별 정확도 분석
    # --------------------------------------
    print("\n📊 클래스별 정확도 분석 중...")
    model.load_state_dict(best_model_wts)
    model.eval()

    class_correct = {cls: 0 for cls in class_names}
    class_total   = {cls: 0 for cls in class_names}

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for label, pred in zip(labels, preds):
                cls = class_names[label.item()]
                class_total[cls] += 1
                if label == pred:
                    class_correct[cls] += 1

    print("\n" + "="*45)
    print(f"{'클래스':<15} {'정답':>5} {'전체':>5} {'정확도':>8}")
    print("-"*45)

    sorted_classes = sorted(class_names,
                            key=lambda c: class_correct[c] / class_total[c] if class_total[c] > 0 else 0)

    for cls in sorted_classes:
        total   = class_total[cls]
        correct = class_correct[cls]
        acc = (correct / total * 100) if total > 0 else 0
        bar = "🟩" * int(acc // 20) + "⬜" * (5 - int(acc // 20))
        print(f"{cls:<15} {correct:>5} {total:>5} {acc:>7.1f}% {bar}")

    print("="*45)

    # --------------------------------------
    # 10. 결과 저장
    # --------------------------------------
    time_elapsed = time.time() - since
    print(f'\n🏆 최종 최고 성능 → Acc: {best_acc*100:.2f}%  Loss: {best_loss_for_log:.4f}')
    print(f'⏱️  소요 시간: {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초')

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, SAVE_PATH)
    print(f"💾 모델 저장 완료: {SAVE_PATH}")
    print(f"📝 학습 로그 저장 완료: {LOG_PATH}")

if __name__ == '__main__':
    main()
