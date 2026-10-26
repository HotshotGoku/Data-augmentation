#!/bin/bash
#SBATCH --job-name=optuna_gensweep
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# P4 Part 2: time-boxed Optuna sweep of SD1.5 branching gen HP, probe-steered. One GPU job, no retrain.
# Submit: sbatch --export=ALL,SWEEP_CKPT=<fixed ckpt>[,N_TRIALS=15] experiments/optuna_gen_sweep_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$REPO/pipeline:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export SWEEP_CKPT=${SWEEP_CKPT:?set SWEEP_CKPT} N_TRIALS=${N_TRIALS:-15}
echo "=== optuna gen sweep ckpt=$SWEEP_CKPT trials=$N_TRIALS $(date) ==="
python experiments/optuna_gen_sweep.py; rc=$?
echo "=== done $(date) exit=$rc ==="
