#!/bin/bash
#SBATCH --job-name=flux_quant
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Flux pilot Stage 1: fp8 quantized inference at 1024. FLUX.1-dev is cached -> run offline (no token).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "=== flux quant smoke start $(date) HW=${HW:-1024} OFFLOAD=${OFFLOAD:-model} ==="
HW=${HW:-1024} STEPS=${STEPS:-28} OFFLOAD=${OFFLOAD:-model} python flux_quant_smoke.py
echo "=== done $(date) exit=$? ==="
