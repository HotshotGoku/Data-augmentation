#!/bin/bash
#SBATCH --job-name=rat_bo
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Bayesian hyperparameter optimization (Optuna TPE) for the Rattray fine-tune.
# In-memory trials, zero checkpoints; resumable SQLite study. Set trial count via BO_TRIALS.
#   sbatch --export=ALL,BO_TRIALS=25 Slurm_scripts/bo_rattray_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
echo "=== rattray BO start $(date) | host $(hostname) | BO_TRIALS=${BO_TRIALS:-25} ==="
python optuna_rattray_bo.py
status=$?
echo "=== rattray BO done $(date) exit $status ==="
exit $status
