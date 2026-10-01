#!/bin/bash
#SBATCH --job-name=flux_cnconf_gen
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Confirm the conditioning-scale diversity win at FULL scale: gen cn0.7 + cn0.6 on all 28 sources x4.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
CKPT=$REPO/flux_cn_runs/branching_v2/controlnet_final.pth
export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch; cd "$REPO"
for s in 0.7 0.6; do
  tag=cn$(echo $s | tr -d .)_full
  echo "=== gen $tag (cn_scale=$s) $(date +%H:%M:%S) ==="
  CN_CKPT="$CKPT" GEN_OUT="$REPO/flux_cn_runs/$tag" CN_LAYERS=2 RES=256 NUM_SAMPLES=4 MAX_SRC=0 \
    CN_SCALE=$s GUID=3.5 STEPS=28 OFFLOAD=none python flux_gen_eval.py || echo "  FAILED $tag"
done
echo "=== gen DONE $(date) ==="
