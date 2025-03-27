import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# === 설정 ===
image_dir = "../train_images"  # 이미지가 들어 있는 폴더
output_csv = "../manual_labels.csv"  # 저장할 CSV
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

# === 이미지 리스트 수집 ===
image_list = sorted([
    f for f in os.listdir(image_dir)
    if os.path.splitext(f)[1].lower() in valid_extensions
])

# === 기존 라벨 불러오기 (있으면 이어서 작업) ===
if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
    labeled_images = set(df["image_name"])
else:
    df = pd.DataFrame(columns=["image_name", "score"])
    labeled_images = set()

# === 라벨링 시작 ===
for image_name in image_list:
    if image_name in labeled_images:
        continue  # 이미 점수 매긴 이미지 스킵

    image_path = os.path.join(image_dir, image_name)
    img = Image.open(image_path)

    # 이미지 표시
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{image_name} - 점수(1~10)를 입력하고 Enter")
    plt.show()

    while True:
        try:
            score = int(input("점수 입력 (1~10): "))
            if 1 <= score <= 10:
                break
            else:
                print("1에서 10 사이의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")

    # 결과 저장
    df = pd.concat([df, pd.DataFrame([{"image_name": image_name, "score": score}])], ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"{image_name} → {score} 저장 완료\n")

print("모든 이미지 라벨링이 완료되었습니다!")
