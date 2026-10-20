#!/bin/bash
#SBATCH --job-name=reptorep_train_sa603
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Training job for the replicate-to-replicate ControlNet augmenter (reptorep_train.py).
# Adapted from Kinshuk's Slurm_scripts/ControlNet_exp_reptorep.sh for NetID sa603.
#
# What it does: trains 5 epochs on the FULL 3-dataset JSON (~300k records) and writes
# YOUR OWN checkpoint to lightning_logs/version_<N>/checkpoints/  -> use that for inference.
# This is a long run (many hours to a few days on a single A5000). Lightning saves a
# checkpoint at the end of each epoch, so partial progress survives a crash.
#
# Submit with:   sbatch Slurm_scripts/train_reptorep_sa603.sh   (from the repo root)
# Watch with:    squeue -u sa603   /   tail -f slurm_logs/reptorep_train_sa603-<jobid>.out

set -uo pipefail

REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction

# --- Conda ---
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction

# --- Paths the aug code needs (matches the verified working recipe) ---
# PHYSICS_DL_PROJECT_PATH lets shared_resources_config.py resolve cldm, control_sd15_ini.ckpt
# and cldm_v15.yaml out of the sim project instead of Kinshuk's inaccessible /hpc/dctrl path.
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
# repo (for `import reptorep_dataset`) : parent (for `Data_augmentation.utils` package) : utils (for top-level `import shared_resources_config`)
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"

# --- Diagnostics + fail-fast GPU check (appear at the top of the .out log) ---
echo "=== start $(date) | host $(hostname) | job ${SLURM_JOB_ID:-?} ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Abort immediately if torch can't init CUDA, instead of dying inside trainer.fit() 90s later.
python -c "import torch, sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, '| cuda avail', ok); sys.exit(0 if ok else 1)" \
  || { echo 'FATAL: CUDA not available to torch on this node — aborting before wasting the allocation.'; exit 1; }

# --- Train ---
python reptorep_train.py
status=$?
echo "=== done $(date) | exit $status ==="
exit $status
