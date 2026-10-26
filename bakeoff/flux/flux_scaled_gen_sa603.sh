#!/bin/bash
#SBATCH --job-name=flux_scaled_gen
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Scaled-Flux generation: 4/source from the 28 held-out test sources at cn1.0 (realism-max) — the
# decisive realism test vs in-domain SD1.5. CN_LAYERS must match training (4). Gen only; score separately.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
CKPT=${CKPT:-$REPO/flux_cn_runs/scaled/controlnet_final.pth}
[ -f "$CKPT" ] || { echo "FATAL: missing $CKPT"; exit 1; }
GEN_OUT=${GEN_OUT:-$REPO/flux_cn_runs/scaled/eval_cn10}   # override per-run so fp8 and bf16 don't clobber
echo "=== scaled-flux gen cn1.0 ckpt=$CKPT CN_LAYERS=${CN_LAYERS:-4} QUANT=${QUANT:-fp8} -> $GEN_OUT $(date) ==="
CN_CKPT="$CKPT" GEN_OUT="$GEN_OUT" CN_LAYERS=${CN_LAYERS:-4} QUANT=${QUANT:-fp8} RES=256 NUM_SAMPLES=4 \
  STEPS=28 GUID=3.5 CN_SCALE=1.0 python bakeoff/flux/flux_gen_eval.py
echo "=== gen done $(date) -> $GEN_OUT ==="
