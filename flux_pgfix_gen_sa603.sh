#!/bin/bash
#SBATCH --job-name=flux_pgfix
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# PG-fixed sweep (gen only): repulsion now scaled to a fraction of latent magnitude. alpha in {0.1,0.3,0.5}.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
CKPT=$REPO/flux_cn_runs/branching_v2/controlnet_final.pth; ROOT=$REPO/flux_cn_runs/pgfix
export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch; cd "$REPO"
for a in 0.1 0.3 0.5 1.0; do
  tag=pgf$(echo $a | tr -d .)
  echo "=== gen $tag (pg_alpha=$a, fixed scaling) $(date +%H:%M:%S) ==="
  CN_CKPT="$CKPT" GEN_OUT="$ROOT/$tag" METHOD=pg PG_ALPHA=$a CN_LAYERS=2 RES=256 NUM_SAMPLES=4 MAX_SRC=8 \
    STEPS=28 GUID=3.5 CN_SCALE=1.0 OFFLOAD=none python flux_gen_diverse.py || echo "  FAILED $tag"
done
echo "=== gen DONE $(date) ==="
