#!/bin/bash
#SBATCH --job-name=mux_flux_gen
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Flux mux generation (cn1.0, realism-max) from mux sources. Separate from score (Flux-in-process segfault).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
CKPT=$REPO/flux_cn_runs/mux/controlnet_final.pth
[ -f "$CKPT" ] || { echo "FATAL: missing $CKPT"; exit 1; }
GEN_OUT=$REPO/flux_cn_runs/mux/eval_cn10
echo "=== flux mux gen cn1.0 -> $GEN_OUT $(date) ==="
CN_CKPT="$CKPT" GEN_OUT="$GEN_OUT" CN_LAYERS=4 QUANT=fp8 RES=256 NUM_SAMPLES=4 \
  STEPS=28 GUID=3.5 CN_SCALE=1.0 SOURCES=$MPX/sources python flux_gen_eval.py
echo "=== gen done $(date) -> $GEN_OUT ==="
