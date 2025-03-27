import torch
import torch.nn as nn
from torchvision.models import resnet18
import os


class NIMA(nn.Module):
    def __init__(self, num_scores=10):
        super(NIMA, self).__init__()

        self.base_model = resnet18(weights=None)  # 사전 학습 안 함
        self.base_model.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_scores),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        return self.head(x)


def load_nima_with_weights(weight_path: str):
    model = NIMA()
    resnet_state_dict = torch.load(weight_path, map_location='cpu')

    # fc.weight, fc.bias 무시하도록 strict=False
    model.base_model.load_state_dict(resnet_state_dict, strict=False)

    return model
