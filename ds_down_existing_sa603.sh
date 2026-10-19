#!/bin/bash
#SBATCH --job-name=down_existing
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu
# Downstream only (no gen) against an EXISTING synth dir. sbatch --export=ALL,DS=<rattray|kl>,BACKBONE=<name>
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction; cd "$REPO"
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
case "${DS:?set DS}" in
  rattray) export REALS="$D/external_datasets/rattray_2023/processed/*.png" LABEL_RE='PIL(?P<lab>[0-9]+)-(?P<rep>[0-9]+)' KIND=cls LOGRATIO=0 TEST_REP=1 ;;
  kl)      export REALS="$D/ood_kl_2species/*.TIF" LABEL_RE='(?P<lab>[0-9]+-[0-9]+)_(?P<rep>[0-9]+)\.TIF' KIND=reg LOGRATIO=1 TEST_REP=1 ;;
  *) echo "unknown DS"; exit 1 ;;
esac
SYN="$REPO/${DS}_synth_${BACKBONE:?set BACKBONE}"
echo "synth coverage: $(ls "$SYN" 2>/dev/null | wc -l) files"
export K=1 N_PER=8 SEEDS=15 OUT="$REPO/${DS}_downstream_${BACKBONE}.tsv"
rm -f "$OUT"
MODE=real      TAG="$DS"              python ds_downstream.py
MODE=rotation  TAG="$DS"              python ds_downstream.py
MODE=augmenter TAG="${DS}_${BACKBONE}" SYNTH="$SYN" python ds_downstream.py
echo "=== $DS/$BACKBONE SUMMARY $(date) ==="; python ds_analyze.py "$OUT"
echo "=== DONE $(date) ==="
