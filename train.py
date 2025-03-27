import torch
import torch.nn as nn
import torch.optim as optim
from models.nima_regression import NIMARegression
from huggingface.load_manual_dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NIMARegression().to(device)
train_loader = get_dataloader()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader.dataset):.4f}")

torch.save(model.state_dict(), "models/nima_finetuned.pth")
