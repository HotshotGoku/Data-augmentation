#!/bin/bash
#SBATCH --job-name=sim_gate
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction; cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export IMAGES="/hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed_100AUG/*.TIF"
export LABEL_RE='(?P<lab>[0-9]+)_(?P<rep>[0-9]+)_rot' KIND=cls MAX_PER_LABEL=40 EPOCHS=25 TAG=sim_condition
echo "=== sim learnability gate (24-way condition from exp images) $(date) ==="
python learnability_gate.py
echo "=== done $(date) ==="
