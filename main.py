import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset import *
from models.model import *

# 데이터셋 및 데이터로더 생성 (예: 'train.csv'와 'train_images' 경로 사용)
train_dataset = AVADataset(csv_file='train.csv', img_dir='train_images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
