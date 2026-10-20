#!/bin/bash
#SBATCH --job-name=ds_prep
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Stage downstream data + generate synthetic replicates (background-black).
# Smoke:  sbatch --export=ALL,DS_SMOKE=1 Slurm_scripts/downstream_prep_sa603.sh
# Full:   sbatch --export=ALL,DS_N=50,DS_M=20 Slurm_scripts/downstream_prep_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
echo "=== ds_prep start $(date) | DS_N=${DS_N:-def} DS_M=${DS_M:-def} DS_SMOKE=${DS_SMOKE:-0} ==="
python downstream_prep.py
status=$?
echo "=== ds_prep done $(date) exit $status ==="
exit $status
