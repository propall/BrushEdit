import os
import torch
from huggingface_hub import snapshot_download

from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler



torch_dtype = torch.float16
device = "cpu"

BrushEdit_path = "models/"
if not os.path.exists(BrushEdit_path):
    BrushEdit_path = snapshot_download(
        repo_id="TencentARC/BrushEdit",
        local_dir=BrushEdit_path,
        token=os.getenv("HF_TOKEN"),
    )
brushnet_path = os.path.join(BrushEdit_path, "brushnetX")
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch_dtype)


base_models_list = [
    # {
    #     "name": "dreamshaper_8 (Preload)",
    #     "local_path": "models/base_model/dreamshaper_8",
    #     "pipe": StableDiffusionBrushNetPipeline.from_pretrained(
    #         "models/base_model/dreamshaper_8", brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
    #     ).to(device)
    # },
    # {
    #     "name": "epicrealism (Preload)",
    #     "local_path": "models/base_model/epicrealism_naturalSinRC1VAE",
    #     "pipe": StableDiffusionBrushNetPipeline.from_pretrained(
    #         "models/base_model/epicrealism_naturalSinRC1VAE", brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
    #     ).to(device)
    # },
    {
        "name": "henmixReal (Preload)",
        "local_path": "models/base_model/henmixReal_v5c",
        "pipe": StableDiffusionBrushNetPipeline.from_pretrained(
            "models/base_model/henmixReal_v5c", brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
        ).to(device)
    },
    {
        "name": "meinamix (Preload)",
        "local_path": "models/base_model/meinamix_meinaV11",
        "pipe": StableDiffusionBrushNetPipeline.from_pretrained(
            "models/base_model/meinamix_meinaV11", brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
        ).to(device)
    },
    {
        "name": "realisticVision (Default)",
        "local_path": "models/base_model/realisticVisionV60B1_v51VAE",
        "pipe": StableDiffusionBrushNetPipeline.from_pretrained(
            "models/base_model/realisticVisionV60B1_v51VAE", brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
        ).to(device)
    },
]

base_models_template = {k["name"]: (k["local_path"], k["pipe"]) for k in base_models_list}
