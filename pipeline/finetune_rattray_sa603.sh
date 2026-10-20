#!/bin/bash
#SBATCH --job-name=ft_rattray
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Fine-tune the augmenter on Rattray P. aeruginosa replicate pairs.
# Sweep example:
#   for lr in 1e-6 5e-6 1e-5; do for fz in none shallow; do
#     sbatch --job-name=ft_rat_${fz}_$lr \
#       --export=ALL,FT_LR=$lr,FT_FREEZE=$fz,FT_EPOCHS=8,FT_TAG=rat_${fz}_$lr,FT_SAVE_TOPK=1 \
#       Slurm_scripts/finetune_rattray_sa603.sh
#   done; done
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
echo "=== ft_rattray start $(date) | tag=${FT_TAG:-?} lr=${FT_LR:-?} freeze=${FT_FREEZE:-?} epochs=${FT_EPOCHS:-?} ==="
python reptorep_finetune_rattray.py
status=$?
echo "=== ft_rattray done $(date) exit $status ==="
exit $status
