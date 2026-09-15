#!/bin/bash
#SBATCH --job-name=rat_bo
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Bayesian HP optimization with PROCESS ISOLATION: each trial runs in a FRESH python process
# (BO_TRIALS=1 per invocation) so GPU memory is fully reclaimed between trials — fixes the OOM
# accumulation seen when running many trials in one process. Shared resumable SQLite study.
#   sbatch --export=ALL,BO_ITERS=20 Slurm_scripts/bo_rattray_loop_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128   # reduce fragmentation OOMs
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

ITERS=${BO_ITERS:-20}
echo "=== rattray BO (process-isolated) start $(date) | host $(hostname) | ITERS=$ITERS ==="
for i in $(seq 1 "$ITERS"); do
  echo "----- trial process $i/$ITERS ($(date +%H:%M:%S)) -----"
  BO_TRIALS=1 python optuna_rattray_bo.py || echo "  (trial process $i errored; continuing)"
done
echo "=== rattray BO done $(date) ==="
