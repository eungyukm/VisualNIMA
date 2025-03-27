from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("eungyukm/image-quality-nima")

# 첫 번째 샘플 출력
sample = dataset["train"][0]
print(sample)

# 이미지 시각화
sample["image"].show()