import os
from PIL import Image
import torch
from models.model import load_nima_with_weights
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경로 설정
project_root = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(project_root, "..", "train_images")
weight_path = os.path.join(project_root, "..", "models", "weights", "resnet18-f37072fd.pth")

# 이미지 확장자 필터
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 모델 로드
model = load_nima_with_weights(weight_path)
model.eval().to(device)

# 이미지 평가 함수
def predict_image_quality(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)[0]

    scores = torch.arange(1, 11).float().to(device)
    mean_score = (output * scores).sum().item()
    return round(mean_score, 2)

# 이미지 파일 리스트 가져오기 (필터링)
image_names = sorted([
    f for f in os.listdir(image_dir)
    if os.path.splitext(f)[1].lower() in valid_extensions
])

# 점수 예측
results = []
for image_name in image_names:
    image_path = os.path.join(image_dir, image_name)
    try:
        score = predict_image_quality(image_path)
        results.append([image_name, score])
        print(f"{image_name} → {score}")
    except Exception as e:
        print(f"{image_name} 처리 중 오류: {e}")

# CSV 저장
df = pd.DataFrame(results, columns=["image_name", "score"])
csv_path = os.path.join(project_root, "..", "train.csv")
df.to_csv(csv_path, index=False)
print(f"\n결과 저장 완료: {csv_path}")

# 히스토그램 시각화
plt.figure(figsize=(10, 5))
plt.hist(df["score"], bins=20, color="skyblue", edgecolor="black")
plt.title("Image Aesthetic Score Distribution")
plt.xlabel("Score")
plt.ylabel("Number of Images")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(project_root, "..", "score_distribution.png"))
plt.show()

# 상위 & 하위 20개 출력
print("\n상위 20 이미지:")
print(df.sort_values(by="score", ascending=False).head(20))

print("\n하위 20 이미지:")
print(df.sort_values(by="score", ascending=True).head(20))
