#!/bin/bash
#SBATCH --job-name=reptorep_eval_sa603
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=0:45:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Quantitative eval of the generated replicates: FID + LPIPS diversity/realism.
# Same env + exports as the training/inference scripts. Reads the latest inference/v*_REPTOREP
# run by default; writes eval_metrics_results/{metrics.json, per_prefix.csv} in the repo.
#
# Submit with:  sbatch Slurm_scripts/eval_metrics_sa603.sh
# First run downloads InceptionV3 + AlexNet weights (needs internet — the youlab-gpu nodes have it).

set -uo pipefail

REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction

export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"

echo "=== start $(date) | host $(hostname) | job ${SLURM_JOB_ID:-?} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch',torch.__version__,'| cuda avail',ok); sys.exit(0 if ok else 1)" \
  || { echo 'FATAL: CUDA not available to torch on this node — aborting.'; exit 1; }

python eval_metrics.py
status=$?
echo "=== done $(date) | exit $status ==="
exit $status
