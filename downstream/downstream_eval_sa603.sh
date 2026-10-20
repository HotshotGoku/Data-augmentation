#!/bin/bash
#SBATCH --job-name=ds_eval
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Paired eval of a trained sim->exp checkpoint on the held-out real test set.
#   sbatch --export=ALL,DS_EVAL_CKPT=<ckpt>,DS_EVAL_TAG=<tag> Slurm_scripts/downstream_eval_sa603.sh
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
echo "=== ds_eval start $(date) | tag=${DS_EVAL_TAG:-?} ckpt=${DS_EVAL_CKPT:-?} ==="
python simtoexp_eval_ds.py
status=$?
echo "=== ds_eval done $(date) exit $status ==="
exit $status
