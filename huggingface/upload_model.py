from huggingface_hub import HfApi, HfFolder, create_repo, upload_file
import os

# Hugging Face 설정
model_name = "nima-regression-finetuned"
username = "eungyukm"
full_repo_name = f"{username}/nima_finetuned"

# 모델 경로 (VisualNIMA 루트 기준)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(root_dir, "models")
model_path = os.path.join(model_dir, "nima_finetuned.pth")

# 모델 파일 존재 확인
assert os.path.exists(model_path), f"모델 파일이 존재하지 않습니다: {model_path}"

# Hugging Face token 확인
token = HfFolder.get_token()
if token is None:
    raise ValueError("Hugging Face 토큰이 설정되어 있지 않습니다. 'huggingface-cli login'을 먼저 실행하세요.")

# 저장소 생성 시도
api = HfApi()
try:
    api.create_repo(name="nima_finetuned", token=token, private=False)
    print(f"✅ 저장소 생성 완료: {full_repo_name}")
except Exception as e:
    print(f"⚠️ 저장소 생성 중 예외 발생 (이미 존재할 수도 있음): {e}")

# 모델 파일 업로드
upload_file(
    path_or_fileobj=model_path,
    path_in_repo="nima_finetuned.pth",
    repo_id=full_repo_name,
    token=token,
)

print(f"✅ 모델 업로드 완료: https://huggingface.co/{full_repo_name}/blob/main/nima_finetuned.pth")
