#!/bin/bash
#SBATCH --job-name=flux_scaled
#SBATCH -p gpu-common
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Scale the Flux ControlNet to give it a fair shot vs in-domain SD1.5 (realism 0.324/CMMD 3.38).
# Data is 256px native -> scale CAPACITY (CN_LAYERS 2->6) + TRAINING (15k->20k), not resolution.
# fp8 base kept (simplest); escalate to bf16 base / H200 if capacity alone doesn't close the gap.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "=== flux SCALED train start $(date) CN_LAYERS=${CN_LAYERS:-6} STEPS=${MAX_STEPS:-20000} RES=${RES:-256} ==="
DATA_JSON=/hpc/group/youlab/sa603/data/branching_256/train_branching.json \
  MAX_PAIRS=0 MAX_STEPS=${MAX_STEPS:-20000} RES=${RES:-256} CN_LAYERS=${CN_LAYERS:-6} LR=${LR:-1e-5} \
  ACCUM=${ACCUM:-1} SAVE_EVERY=${SAVE_EVERY:-5000} OUT=$REPO/flux_cn_runs/${RUN:-scaled} \
  python flux_train_controlnet.py
echo "=== done $(date) exit=$? ==="; ls -la $REPO/flux_cn_runs/${RUN:-scaled}/*.pth 2>/dev/null
