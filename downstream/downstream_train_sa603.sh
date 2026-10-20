#!/bin/bash
#SBATCH --job-name=ds_train
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Train sim->exp from the ControlNet init on a given JSON (real-only OR real+synthetic).
#   sbatch --export=ALL,DS_JSON=<path>,DS_TAG=<tag> Slurm_scripts/downstream_train_sa603.sh
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
echo "=== ds_train start $(date) | tag=${DS_TAG:-?} json=${DS_JSON:-?} ==="
python simtoexp_train_ds.py
status=$?
echo "=== ds_train done $(date) exit $status ==="
exit $status
