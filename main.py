import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from dataset.dataset import *
from models.model import *

# 데이터셋 및 데이터로더 생성 (예: 'train.csv'와 'train_images' 경로 사용)
train_dataset = AVADataset(csv_file='train.csv', img_dir='train_images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델 인스턴스 생성
model = NIMA(num_scores=10)  # 예시로 10개의 점수 클래스

# GPU 사용 여부 확인 (있으면 GPU로 모델을 이동)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()  # 클래스 분류 문제의 경우
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 10
model.train()  # 학습 모드 설정
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # 배치 데이터를 GPU로 이동
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
