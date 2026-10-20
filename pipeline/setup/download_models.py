"""Pre-stage screen backbones into HF cache on the login node (has internet; compute runs offline)."""
import os
os.environ.setdefault("HF_HOME", "/hpc/group/youlab/sa603/.hf_home")
from huggingface_hub import snapshot_download
MODELS = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
]
for m in MODELS:
    print("DOWNLOADING", m, flush=True)
    try:
        snapshot_download(m, ignore_patterns=["*.onnx", "*onnx*", "*non_ema*", "*.ckpt", "*.pth",
                                              "*flax*", "*openvino*", "*.msgpack"])
        print("DONE", m, flush=True)
    except Exception as e:
        print("FAIL", m, repr(e)[:200], flush=True)
print("ALLDONE", flush=True)
