#!/bin/bash
#SBATCH --job-name=mux_flux_train
#SBATCH -p gpu-common
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Fill matrix cell: Flux ControlNet (fp8/CN4) on MULTIPLEXED. Train only; gen+score chained after.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "=== flux MUX train $(date) CN_LAYERS=${CN_LAYERS:-4} STEPS=${MAX_STEPS:-20000} ==="
DATA_JSON=/hpc/group/youlab/sa603/data/multiplexed_eval/train_multiplexed.json \
  MAX_PAIRS=0 MAX_STEPS=${MAX_STEPS:-20000} RES=256 CN_LAYERS=${CN_LAYERS:-4} LR=1e-5 \
  ACCUM=1 SAVE_EVERY=5000 OUT=$REPO/flux_cn_runs/mux \
  python flux_train_controlnet.py
echo "=== done $(date) exit=$? ==="; ls -la $REPO/flux_cn_runs/mux/*.pth 2>/dev/null
