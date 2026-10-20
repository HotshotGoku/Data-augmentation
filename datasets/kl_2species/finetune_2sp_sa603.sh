#!/bin/bash
#SBATCH --job-name=reptorep_ft2sp
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Fine-tune the augmenter on the 2-species TRAIN split, resuming from our full-data model.
# Hyperparameters come from env vars (set via `sbatch --export`); one checkpoint saved per epoch.
#
# Single probe run (defaults: lr 5e-6, 4 epochs):
#   sbatch Slurm_scripts/finetune_2sp_sa603.sh
#
# Learning-rate sweep (each run saves per-epoch ckpts -> epochs axis is free):
#   for lr in 1e-6 5e-6 1e-5; do
#     sbatch --job-name=ft2sp_lr$lr --export=ALL,FT_LR=$lr,FT_EPOCHS=4,FT_TAG=lr$lr \
#            Slurm_scripts/finetune_2sp_sa603.sh
#   done

set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"

echo "=== start $(date) | host $(hostname) | job ${SLURM_JOB_ID:-?} | FT_LR=${FT_LR:-default} FT_EPOCHS=${FT_EPOCHS:-default} FT_TAG=${FT_TAG:-default} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA on this node'; exit 1; }

python reptorep_finetune.py
status=$?
echo "=== done $(date) | exit $status ==="
exit $status
