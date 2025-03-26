import torch.nn as nn
import torchvision.models as models


class NIMA(nn.Module):
    def __init__(self, num_scores=10):
        super(NIMA, self).__init__()
        # 미리 학습된 ResNet50 사용
        self.base_model = models.resnet50(pretrained=True)
        in_features = self.base_model.fc.in_features
        # 마지막 fully connected layer를 미적 평가용으로 수정
        self.base_model.fc = nn.Linear(in_features, num_scores)
        # 출력 후 softmax를 적용하여 확률 분포로 만듦
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.softmax(x)
        return x


# 모델 인스턴스 생성
model = NIMA(num_scores=10)