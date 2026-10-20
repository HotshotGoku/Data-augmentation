#!/bin/bash
#SBATCH --job-name=flux_bf16
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Last Flux lever: bf16 base (QUANT=none, no fp8 quality loss) + CN6 on A6000-48GB.
# scavenger is preemptible -> --requeue + auto-resume from the latest saved checkpoint.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
OUT=$REPO/flux_cn_runs/${RUN:-scaled_bf16}
LATEST=$(ls -t "$OUT"/controlnet_step*.pth 2>/dev/null | head -1)
[ -n "${LATEST:-}" ] && export RESUME_CKPT="$LATEST" && echo "resume from $LATEST"
echo "=== flux bf16 train QUANT=none CN_LAYERS=${CN_LAYERS:-6} STEPS=${MAX_STEPS:-20000} $(date) ==="
QUANT=none DATA_JSON=/hpc/group/youlab/sa603/data/branching_256/train_branching.json \
  MAX_PAIRS=0 MAX_STEPS=${MAX_STEPS:-20000} RES=256 CN_LAYERS=${CN_LAYERS:-6} LR=1e-5 \
  ACCUM=1 SAVE_EVERY=${SAVE_EVERY:-2500} OUT="$OUT" \
  python flux_train_controlnet.py
echo "=== done $(date) exit=$? ==="; ls -la "$OUT"/*.pth 2>/dev/null | tail -3
