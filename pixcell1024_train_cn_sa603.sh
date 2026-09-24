#!/bin/bash
#SBATCH --job-name=pc1024_cn
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# PixCell-1024 ControlNet fine-tune on a 48GB scavenger GPU (A6000/6000-Ada).
# Smoke: sbatch --export=ALL,MAX_STEPS=20 ...  ; Full: no MAX_STEPS.
# --requeue + auto-resume from latest checkpoint = survives scavenger preemption.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
# TAG selects the dataset: mplex (256->1024 upscaled, validation) | branching (native 1001->1024, real test)
TAG=${TAG:-mplex}
case "$TAG" in
  branching) DEF_JSON=/hpc/group/youlab/sa603/data/branching_1024/train_branching.json ;;
  *)         DEF_JSON=/hpc/group/youlab/sa603/data/multiplexed_eval/train_multiplexed.json ;;
esac
TRAIN_JSON=${TRAIN_JSON:-$DEF_JSON}
OUT=${OUT:-$REPO/pixcell1024_cn_runs/$TAG}
mkdir -p "$OUT"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export PYTHONPATH=/hpc/group/youlab/sa603/code/PixCell/controlnet:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1  # live step logs (else stdout block-buffers to the .out file)
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/pixcell
cd "$REPO"
echo "=== node/GPU $(date) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

# auto-resume from the latest saved checkpoint (preemption-safe)
RESUME=""; latest=$(ls -t "$OUT"/controlnet_step*.pth 2>/dev/null | head -1); [ -n "$latest" ] && RESUME="$latest"
echo "=== train start $(date) TAG=$TAG TRAIN_JSON=$TRAIN_JSON MAX_STEPS=${MAX_STEPS:-0} RESUME=${RESUME:-none} ==="
OUT="$OUT" TRAIN_JSON="$TRAIN_JSON" RESUME_CKPT="$RESUME" python pixcell1024_train_cn.py
echo "=== exit=$? $(date) ==="; ls -la "$OUT" 2>&1
