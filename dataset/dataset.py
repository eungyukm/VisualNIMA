import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AVADataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # CSV 파일에 "image_name"과 "score" 컬럼이 있다고 가정합니다.
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx]['image_name'])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # score가 1~10의 정수로 분류되는 경우 (클래스 분류)
        score = self.annotations.iloc[idx]['score']
        score = int(score)

        # 연속적인 값이 있을 경우 (회귀 문제로 처리할 때)
        # score = float(score)  # 회귀 문제에서는 이 방식으로 처리할 수 있습니다.

        return image, score


# 이미지 전처리 정의 (모델에 맞게 크기 조정 및 정규화)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])