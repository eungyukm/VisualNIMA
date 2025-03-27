from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

def get_dataloader(batch_size=32):
    dataset = load_dataset("eungyukm/image-manual-label")["train"]

    # 필요한 컬럼만 가져오도록 설정
    dataset.set_format(type="python", columns=["image", "score", "image_name"])

    # transform 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def collate_fn(batch):
        images = []
        scores = []
        for item in batch:
            image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")  # RGBA → RGB 변환

            tensor = transform(image)
            images.append(tensor)
            scores.append(item["score"])

        return torch.stack(images), torch.tensor(scores, dtype=torch.float32)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
