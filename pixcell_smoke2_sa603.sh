#!/bin/bash
#SBATCH --job-name=pixcell_smoke2
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=1:30:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# PixCell smoke v2: fixed loader (GitHub controlnet/ on PYTHONPATH) + bundled VAE + null uni token.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export OUT=$REPO/pixcell_smoke_out
mkdir -p "$OUT"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export PYTHONPATH=/hpc/group/youlab/sa603/code/PixCell/controlnet:${PYTHONPATH:-}
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/pixcell
python -c "import torch,sys; print('cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
python "$REPO/pixcell_smoke2.py"
echo "=== done $(date) ==="; ls -la "$OUT"
