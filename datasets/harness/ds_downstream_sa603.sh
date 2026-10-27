#!/bin/bash
#SBATCH --job-name=ds_downstream
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# 4-arm downstream at SEEDS seeds against EXISTING synth dirs (no gen).
# sbatch --export=ALL,DS=<kl|rattray>[,SEEDS=15] ds_downstream_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home

case "${DS:?set DS}" in
  rattray) export REALS="$D/clean/external_datasets/rattray_2023/processed/*.png" \
           LABEL_RE='PIL(?P<lab>[0-9]+)-(?P<rep>[0-9]+)' KIND=cls LOGRATIO=0 TEST_REP=1 \
           SYIN="$D/clean/external_datasets/rattray_2023/synth_clean" SYDR="$D/clean/external_datasets/rattray_2023/synth_clean_dropout" ;;
  kl)      export REALS="$D/clean/ood_kl_2species/*.TIF" \
           LABEL_RE='(?P<lab>[0-9]+-[0-9]+)_(?P<rep>[0-9]+)\.TIF' KIND=reg LOGRATIO=1 TEST_REP=1 \
           SYIN="$D/clean/ood_kl_2species/synth_clean" SYDR="$D/clean/ood_kl_2species/synth_clean_dropout" ;;
  *) echo "unknown DS=$DS"; exit 1 ;;
esac
export K=1 N_PER=8 SEEDS=${SEEDS:-15} BACKBONE=${BACKBONE:-resnet50} EPOCHS=${EPOCHS:-30}
export OUT="$REPO/downstream/outputs/${DS}_downstream_p5.tsv"; mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
echo "=== $DS downstream SEEDS=${SEEDS:-15} backbone=$BACKBONE $(date) ==="
MODE=real      TAG="$DS"            python datasets/harness/ds_downstream.py
MODE=real_dup  TAG="$DS"            python datasets/harness/ds_downstream.py
MODE=rotation  TAG="$DS"            python datasets/harness/ds_downstream.py
MODE=augmenter TAG="${DS}_indomain" SYNTH="$SYIN" python datasets/harness/ds_downstream.py
MODE=augmenter TAG="${DS}_dropout"  SYNTH="$SYDR" python datasets/harness/ds_downstream.py
MODE=real      TAG="${DS}_ceiling" K=999 python datasets/harness/ds_downstream.py
echo "=== $DS SUMMARY (SEEDS=${SEEDS:-15}) $(date) ==="; python datasets/harness/ds_analyze.py "$OUT"
echo "=== DONE $(date) ==="
