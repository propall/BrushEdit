import os
from huggingface_hub import snapshot_download

# download hf models
BrushEdit_path = "models/"
if not os.path.exists(BrushEdit_path):
    BrushEdit_path = snapshot_download(
        repo_id="TencentARC/BrushEdit",
        local_dir=BrushEdit_path,
        token=os.getenv("HF_TOKEN"),
    )

print("Downloaded BrushEdit to ", BrushEdit_path)
