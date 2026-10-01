#!/bin/bash
#SBATCH --job-name=flux_overfit
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Flux pilot Stage 2: overfit a few branching pairs to confirm the flow-matching CN loop learns.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "=== flux overfit start $(date) ==="
DATA_JSON=/hpc/group/youlab/sa603/data/branching_256/train_branching.json \
  MAX_PAIRS=${MAX_PAIRS:-8} MAX_STEPS=${MAX_STEPS:-300} RES=${RES:-512} ACCUM=${ACCUM:-1} \
  CN_LAYERS=${CN_LAYERS:-4} SAVE_EVERY=1000 OUT=$REPO/flux_cn_runs/overfit \
  python flux_train_controlnet.py
echo "=== done $(date) exit=$? ==="
