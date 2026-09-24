#!/bin/bash
#SBATCH --job-name=pixcell_cn
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Fine-tune the PixCell-256 Cell-ControlNet (warm-start released CN, train ControlNet only).
# TAG selects the dataset: mplex (default) | branching. Override TRAIN_JSON/OUT via --export if desired.
# Smoke: sbatch --export=ALL,MAX_STEPS=50,EPOCHS=1 ; Full: no MAX_STEPS (EPOCHS defaults to 40).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
TAG=${TAG:-mplex}
case "$TAG" in
  branching) DEF_JSON=/hpc/group/youlab/sa603/data/branching_256/train_branching.json ;;
  *)         DEF_JSON=/hpc/group/youlab/sa603/data/multiplexed_eval/train_multiplexed.json ;;
esac
export TRAIN_JSON=${TRAIN_JSON:-$DEF_JSON}
export OUT=${OUT:-$REPO/pixcell_cn_runs/$TAG}
mkdir -p "$OUT"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export PYTHONPATH=/hpc/group/youlab/sa603/code/PixCell/controlnet:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/pixcell
cd "$REPO"
python -c "import torch,sys; print('cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
echo "=== pixcell-cn train start $(date) TAG=$TAG TRAIN_JSON=$TRAIN_JSON OUT=$OUT MAX_STEPS=${MAX_STEPS:-0} EPOCHS=${EPOCHS:-40} ==="
python pixcell_train_cn.py
echo "=== pixcell-cn exit=$? $(date) ==="
ls -la "$OUT" 2>&1
