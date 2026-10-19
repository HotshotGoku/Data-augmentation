#!/bin/bash
#SBATCH --job-name=ddpm_bake
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu
# Bake-off: train a from-scratch conditional DDPM augmenter on a dataset. sbatch --export=ALL,DS=<rattray|kl>
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/ddpm; cd "$REPO"
case "${DS:?set DS}" in
  rattray) DJ=/hpc/group/youlab/sa603/data/external_datasets/rattray_2023/train_rattray.json ;;
  kl)      DJ=/hpc/group/youlab/sa603/data/ood_kl_2species/train_kl.json ;;
  *) echo "unknown DS"; exit 1 ;;
esac
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo "=== ddpm bakeoff train DS=$DS $(date) ==="
MODE=train DATA_JSON="$DJ" STEPS=${STEPS:-40000} OUT=$REPO/cond_ddpm_runs/${DS} python conditional_ddpm.py
echo "=== done $(date) exit=$? ==="; ls -la $REPO/cond_ddpm_runs/${DS}/*.pth 2>/dev/null
