#!/bin/bash
#SBATCH --job-name=flux_combo2
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Finer combo sweep to clear diversity>0.266 while keeping CMMD<8.8 (cn0.7+PG0.5 had CMMD headroom).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
CKPT=$REPO/flux_cn_runs/branching_v2/controlnet_final.pth; ROOT=$REPO/flux_cn_runs/combo2
export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch; cd "$REPO"
# tag:CN_SCALE:PG_ALPHA
CONFIGS="cn07pg06:0.7:0.6 cn07pg07:0.7:0.7 cn07pg08:0.7:0.8 cn065pg06:0.65:0.6"
for c in $CONFIGS; do
  IFS=: read -r tag scale alpha <<< "$c"
  echo "=== gen $tag (cn_scale=$scale pg_alpha=$alpha) $(date +%H:%M:%S) ==="
  CN_CKPT="$CKPT" GEN_OUT="$ROOT/$tag" METHOD=pg PG_ALPHA=$alpha CN_SCALE=$scale \
    CN_LAYERS=2 RES=256 NUM_SAMPLES=4 MAX_SRC=0 STEPS=28 GUID=3.5 OFFLOAD=none \
    python flux_gen_diverse.py || echo "  FAILED $tag"
done
echo "=== gen DONE $(date) ==="
