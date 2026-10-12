#!/bin/bash
#SBATCH --job-name=bbscreen
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=1-4%2
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%A_%a.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Automated backbone screen: one array task per candidate (generic AutoPipeline img2img). Robust to
# per-model failures (a bad model just fails its own task). Reads candidate N from screen_candidates.txt.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" screen_candidates.txt)
export NAME=$(echo "$LINE" | cut -d'|' -f1)
export MODEL_ID=$(echo "$LINE" | cut -d'|' -f2)
export DTYPE=$(echo "$LINE" | cut -d'|' -f3)
echo "=== [$SLURM_ARRAY_TASK_ID] screen $NAME ($MODEL_ID) $(date) ==="
NUM=4 STEPS=30 STRENGTHS=0.5,0.7 RES=256 OUTROOT=$REPO/backbone_screen python screen_img2img.py
echo "=== done $NAME $(date) exit=$? ==="
