#!/bin/bash
#SBATCH --job-name=flux_bake
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu
# Bake-off: train a Flux ControlNet augmenter on a dataset. sbatch --export=ALL,DS=<rattray|kl> ds_flux_bakeoff_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch; cd "$REPO"
case "${DS:?set DS}" in
  rattray) DJ=/hpc/group/youlab/sa603/data/external_datasets/rattray_2023/train_rattray.json ;;
  kl)      DJ=/hpc/group/youlab/sa603/data/ood_kl_2species/train_kl.json ;;
  *) echo "unknown DS"; exit 1 ;;
esac
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo "=== flux bakeoff train DS=$DS $(date) ==="
DATA_JSON="$DJ" MAX_PAIRS=0 MAX_STEPS=${MAX_STEPS:-15000} RES=256 CN_LAYERS=4 LR=1e-5 ACCUM=1 \
  SAVE_EVERY=5000 OUT=$REPO/flux_cn_runs/${DS} python flux_train_controlnet.py
echo "=== done $(date) exit=$? ==="; ls -la $REPO/flux_cn_runs/${DS}/*.pth 2>/dev/null
