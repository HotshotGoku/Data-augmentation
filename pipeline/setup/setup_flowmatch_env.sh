#!/bin/bash
# Build a modern env for the flow-matching pilot (SD3.5 + ControlNet). In GROUP storage (home is full).
set -e
ENVDIR=/hpc/group/youlab/sa603/envs/flowmatch
export PIP_CACHE_DIR=/hpc/group/youlab/sa603/.pipcache
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh

echo "=== create env $(date) ==="
conda create -y -p "$ENVDIR" python=3.10
conda activate "$ENVDIR"

echo "=== torch (cu121) $(date) ==="
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo "=== diffusers stack $(date) ==="
pip install "diffusers==0.31.0" "transformers==4.44.2" accelerate safetensors sentencepiece protobuf peft "huggingface_hub>=0.25" hf_transfer
pip install opencv-python-headless pillow numpy scipy lpips

echo "=== verify $(date) ==="
python -c "import torch,diffusers,transformers,lpips; print('torch',torch.__version__,'| diffusers',diffusers.__version__,'| transformers',transformers.__version__,'| cuda_build',torch.version.cuda)"
python -c "from diffusers import AutoencoderKL, SD3Transformer2DModel; print('SD3 classes import OK')"
echo "ENV_SETUP_DONE $(date)"
