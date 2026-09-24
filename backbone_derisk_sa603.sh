#!/bin/bash
#SBATCH --job-name=backbone_derisk
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Cheap de-risk gates for both backbones before any training:
#  1) BBDM: does the vq-f4 autoencoder reconstruct colony texture? (env bbdm)
#  2) PixCell-256: does the model+ControlNet load and run with our source-image conditioning? (env pixcell)
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export OUT=$REPO/backbone_derisk_out
mkdir -p "$OUT"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh

echo "############## BBDM VQGAN recon (env bbdm) $(date) ##############"
conda activate /hpc/group/youlab/sa603/envs/bbdm
python -c "import torch,sys; print('cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" || echo "WARN: no CUDA in bbdm env"
python "$REPO/bbdm_vqgan_recon.py" || echo "!! BBDM RECON FAILED"
conda deactivate

echo "############## PixCell zero-shot smoke (env pixcell) $(date) ##############"
conda activate /hpc/group/youlab/sa603/envs/pixcell
python -c "import torch,sys; print('cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" || echo "WARN: no CUDA in pixcell env"
python "$REPO/pixcell_smoke.py" || echo "!! PIXCELL SMOKE FAILED"
conda deactivate

echo "############## DERISK DONE $(date) ##############"
ls -la "$OUT"
