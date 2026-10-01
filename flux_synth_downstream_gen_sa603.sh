#!/bin/bash
#SBATCH --job-name=flux_dsynth
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Generate Flux (cn0.65+PG0.6) synth for the downstream train pool.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch; cd "$REPO"
echo "=== flux downstream synth gen start $(date) ==="
CN_CKPT=$REPO/flux_cn_runs/branching_v2/controlnet_final.pth \
  GEN_OUT=/hpc/group/youlab/sa603/data/branching_downstream/synth_train_flux \
  REAL_DIR=/hpc/group/youlab/sa603/data/branching_256/canon \
  EXCLUDE_DIR=/hpc/group/youlab/sa603/data/branching_256/test/reals \
  K=8 BATCH=4 CN_SCALE=0.65 PG_ALPHA=0.6 CN_LAYERS=2 RES=256 STEPS=28 python flux_gen_synth_downstream.py
echo "=== count: $(ls /hpc/group/youlab/sa603/data/branching_downstream/synth_train_flux/*.png 2>/dev/null | wc -l) ==="
echo "=== DONE $(date) ==="
