#!/bin/bash
#SBATCH --job-name=reptorep_infer_sa603
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Inference for the replicate-to-replicate ControlNet augmenter (reptorep_infer.py).
# Adapted from Kinshuk's Slurm_scripts/ControlNet_exp_reptorep_infer.sh for NetID sa603,
# using the SAME cd + export fix as the verified training script.
#
# PREREQ: edit utils/local_config.py so pipeline.py loads YOUR checkpoint and writes to a
# path you can access (see CKPT_PATH_V4 and OUTPUT_DIR_REPTOREP).
#
# Reads the 3 held-out test folders (EXP_FOLDER_TEST, EXP_FOLDER_TEST_V3, EMRAH_EXP_FOLDER_TEST),
# generates num_samples=2 images per test prefix at 50 DDIM steps, writes PNGs to OUTPUT_DIR_REPTOREP.
# Short run (~15-20 min). --exclusive is the proven-working GPU config on this cluster
# (single-GPU --gres=gpu:1 has failed with a CUDA-init cgroup error here).
#
# Submit with:  sbatch Slurm_scripts/infer_reptorep_sa603.sh   (from the repo root)

set -uo pipefail

REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction

# --- Conda ---
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction

# --- Paths the aug code needs (matches the verified working training recipe) ---
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"

# --- Diagnostics + fail-fast GPU check ---
echo "=== start $(date) | host $(hostname) | job ${SLURM_JOB_ID:-?} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch, sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, '| cuda avail', ok); sys.exit(0 if ok else 1)" \
  || { echo 'FATAL: CUDA not available to torch on this node — aborting.'; exit 1; }

# --- Inference ---
python reptorep_infer.py
status=$?
echo "=== done $(date) | exit $status ==="
exit $status
