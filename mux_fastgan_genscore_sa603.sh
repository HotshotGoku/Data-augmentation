#!/bin/bash
#SBATCH --job-name=mux_lwgan_score
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# FastGAN mux: generate 400 individual images (lwgan env), then direct CMMD + global diversity vs mux reals.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh

conda activate /hpc/group/youlab/sa603/envs/lwgan; cd "$REPO"
echo "=== lwgan mux gen $(date) ==="
NAME=lwgan_mux N=400 OUT=$REPO/gan_runs/gen/lwgan_mux python gan_generate.py

echo "=== direct CMMD vs mux reals (pytorch_PA) $(date) ==="
conda deactivate; conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
G=$REPO/gan_runs/gen/lwgan_mux REALS=$MPX/reals python cmmd_direct.py
echo ">>> FASTGAN (mux) scored above. REF mux: SD1.5 in-domain CMMD 12.6 | PixCell 5.78"
echo "=== DONE $(date) ==="
