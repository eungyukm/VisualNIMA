import torch
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from models.nima_regression import NIMARegression  # 사용자 정의 모델

# 1. 모델 가중치 Hugging Face에서 다운로드
model_path = hf_hub_download(
    repo_id="eungyukm/nima_finetuned",     # 모델 저장소 ID
    filename="nima_finetuned.pth"          # 업로드된 모델 파일명
)

# 2. 이미지 로딩 및 전처리
image = Image.open("example.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

input_tensor = transform(image).unsqueeze(0)

# 3. 모델 로드
model = NIMARegression()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# 4. 예측 수행
with torch.no_grad():
    score = model(input_tensor).item()

print(f"예측된 미적 점수: {score:.2f}")