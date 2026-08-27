#!/bin/bash
#SBATCH --job-name=es_prep
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=6:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Build exp->seed A/B folders + generate augmenter replicates.
# Smoke: sbatch --export=ALL,ES_SMOKE=1 Slurm_scripts/expseed_prep_sa603.sh
# Full:  sbatch --export=ALL,ES_N=2500,ES_M=2 Slurm_scripts/expseed_prep_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
echo "=== es_prep start $(date) | ES_N=${ES_N:-def} ES_M=${ES_M:-def} ES_SMOKE=${ES_SMOKE:-0} ==="
python expseed_prep.py
status=$?
echo "=== es_prep done $(date) exit $status ==="
exit $status
