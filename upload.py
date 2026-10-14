from pathlib import Path
from huggingface_hub import HfApi

token_file = Path("backend/token.txt")
if not token_file.exists():
    token_file = Path("token.txt")
token = token_file.read_text().strip()

api = HfApi(token=token)
user = api.whoami()["name"]
repo = f"{user}/comic-forge"
print("Uploading as", user)

api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(
    folder_path=".",
    repo_id=repo,
    repo_type="space",
    ignore_patterns=["venv/*", "**/venv/*", "**/node_modules/*", ".git/*",
                     ".env", "**/.env", "token.txt", "**/token.txt",
                     "**/storage/*", "**/__pycache__/*"],
)
print("Done: https://huggingface.co/spaces/" + repo)
