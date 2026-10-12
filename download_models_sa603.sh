#!/bin/bash
#SBATCH --job-name=bbdownload
#SBATCH -p common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
echo "=== internet test from compute node ==="; curl -sI --max-time 15 https://huggingface.co 2>&1 | head -1
echo "=== download start $(date) ==="
python download_models.py
echo "=== done $(date) ==="; du -sh "$HF_HOME" 2>/dev/null
