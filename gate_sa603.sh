#!/bin/bash
#SBATCH --job-name=gate
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Phase-0 learnability gate. Submit with: sbatch --export=ALL,DS=<rattray|kl|swarmevo|myxo> gate_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

case "${DS:?set DS}" in
  rattray)  export IMAGES="$D/external_datasets/rattray_2023/processed/*.png" \
            LABEL_RE='PIL(?P<lab>[0-9]+)-(?P<rep>[0-9]+)' KIND=cls TEST_REP=1 TAG=rattray_strain ;;
  kl)       export IMAGES="$D/ood_kl_2species/*.TIF" \
            LABEL_RE='(?P<lab>[0-9]+-[0-9]+)_(?P<rep>[0-9]+)\.TIF' KIND=reg LOGRATIO=1 TEST_REP=1 TAG=kl_ratio ;;
  swarmevo) export IMAGES="$D/external_datasets/swarmevo/canon/**/*.png" \
            LABEL_RE='(?:train|test)_(?P<lab>[0-9]+)/(?P<rep>[0-9]+)\.png' KIND=cls MAX_PER_LABEL=60 TAG=swarmevo_lineage ;;
  myxo)     export IMAGES="$D/external_datasets/myxococcus/images/**/*" \
            LABEL_RE='images/(?P<lab>[A-Za-z0-9_%]+)_Run[0-9]+' KIND=cls TAG=myxo_strain ;;
  *) echo "unknown DS=$DS"; exit 1 ;;
esac
echo "=== gate DS=$DS TAG=$TAG $(date) ==="
python learnability_gate.py
echo "=== done $(date) ==="
