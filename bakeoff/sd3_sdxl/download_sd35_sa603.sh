#!/bin/bash
#SBATCH --job-name=sd35dl
#SBATCH -p common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
set -uo pipefail
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export HF_TOKEN=$(cat "$HOME/.hf_token")
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
echo "=== auth check ==="; huggingface-cli whoami
echo "=== download SD3.5-medium (diffusers) $(date) ==="
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("stabilityai/stable-diffusion-3.5-medium",
    ignore_patterns=["*.onnx","*onnx*","*.ckpt","*.gguf","*q4*","*q8*","*fp8*","*.bin"])
print("snapshot:", p)
import os
print("has model_index.json:", os.path.exists(os.path.join(p, "model_index.json")))
PY
echo "=== done $(date) ==="
