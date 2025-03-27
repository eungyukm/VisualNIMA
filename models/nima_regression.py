import torch
import torch.nn as nn
from torchvision.models import resnet18

class NIMARegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = resnet18(weights="IMAGENET1K_V1")
        self.base_model.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.base_model(x)
        return self.head(x).squeeze(1)
