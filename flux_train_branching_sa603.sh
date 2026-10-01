#!/bin/bash
#SBATCH --job-name=flux_branch
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Flux pilot Stage 3: fine-tune a Flux ControlNet on the full branching domain (validated loop).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "=== flux branching fine-tune start $(date) ==="
DATA_JSON=/hpc/group/youlab/sa603/data/branching_256/train_branching.json \
  MAX_PAIRS=0 MAX_STEPS=${MAX_STEPS:-6000} RES=${RES:-256} CN_LAYERS=${CN_LAYERS:-2} LR=${LR:-1e-5} \
  ACCUM=${ACCUM:-4} SAVE_EVERY=${SAVE_EVERY:-1500} OUT=$REPO/flux_cn_runs/${RUN:-branching} \
  python flux_train_controlnet.py
echo "=== done $(date) exit=$? ==="
