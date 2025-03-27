from datasets import Dataset, Features, Value, Image
import pandas as pd
import os

# === 경로 설정 ===
csv_path = "../train.csv"
image_dir = "../train_images"  # 이미지가 들어 있는 폴더
repo_name = "eungyukm/image-quality-nima"  # 사용자명/리포명으로 수정
private = False  # True면 비공개 업로드

# === CSV 로드 및 이미지 경로 추가 ===
df = pd.read_csv(csv_path)
df["image"] = df["image_name"].apply(lambda x: os.path.join(image_dir, x))

# === 데이터셋 포맷 정의 ===
features = Features({
    "image_name": Value("string"),
    "score": Value("float32"),
    "image": Image()
})

# === Dataset 객체 생성 ===
dataset = Dataset.from_pandas(df, features=features)

# === 이미지 컬럼 형 변환 (필요 시) ===
dataset = dataset.cast_column("image", Image())

# === 업로드 ===
dataset.push_to_hub(repo_name, private=private)

print(f"Hugging Face 업로드 완료: https://huggingface.co/datasets/{repo_name}")
