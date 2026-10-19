#!/bin/bash
#SBATCH --job-name=bb_downstream
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu
# Bake-off downstream: gen per-source synth from a trained alt-backbone augmenter, then 15-seed
# downstream vs rotation. sbatch --export=ALL,DS=<rattray|kl>,BACKBONE=<flux|ddpm> ds_backbone_downstream_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
case "${DS:?set DS}" in
  rattray) SRCDIR="$D/external_datasets/rattray_2023/processed"; SPAT='*.png'; REALS_G="$SRCDIR/*.png"; LRE='PIL(?P<lab>[0-9]+)-(?P<rep>[0-9]+)'; KI=cls; LG=0 ;;
  kl)      SRCDIR="$D/ood_kl_2species"; SPAT='*.TIF'; REALS_G="$SRCDIR/*.TIF"; LRE='(?P<lab>[0-9]+-[0-9]+)_(?P<rep>[0-9]+)\.TIF'; KI=reg; LG=1 ;;
  *) echo "unknown DS"; exit 1 ;;
esac
SYN="$REPO/${DS}_synth_${BACKBONE:?set BACKBONE}"; rm -rf "$SYN"; mkdir -p "$SYN"

echo "=== GEN $BACKBONE on $DS -> $SYN $(date) ==="
case "$BACKBONE" in
  flux)
    conda activate /hpc/group/youlab/sa603/envs/flowmatch; cd "$REPO"
    export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CN_CKPT="$REPO/flux_cn_runs/${DS}/controlnet_final.pth" GEN_OUT="$SYN" SOURCES="$SRCDIR" SRC_PAT="$SPAT" STEM=1 \
      CN_LAYERS=4 QUANT=fp8 RES=256 NUM_SAMPLES=8 STEPS=28 GUID=3.5 CN_SCALE=1.0 python flux_gen_eval.py ;;
  ddpm)
    conda activate /hpc/group/youlab/sa603/envs/ddpm; cd "$REPO"
    MODE=gen CKPT="$REPO/cond_ddpm_runs/${DS}/ema_final.pth" OUT="$SYN" SOURCES="$SRCDIR" SRC_PAT="$SPAT" STEM=1 \
      NUM=8 STEPS_SAMPLE=50 ETA=0.0 python conditional_ddpm.py ;;
  *) echo "unknown BACKBONE"; exit 1 ;;
esac
conda deactivate 2>/dev/null || true

echo "=== DOWNSTREAM 15-seed (real/rotation/aug-$BACKBONE) $(date) ==="
conda activate pytorch_PA_patternprediction; cd "$REPO"
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export REALS="$REALS_G" LABEL_RE="$LRE" KIND="$KI" LOGRATIO="$LG" TEST_REP=1 K=1 N_PER=8 SEEDS=15 OUT="$REPO/${DS}_downstream_${BACKBONE}.tsv"
rm -f "$OUT"
MODE=real      TAG="$DS"              python ds_downstream.py
MODE=rotation  TAG="$DS"              python ds_downstream.py
MODE=augmenter TAG="${DS}_${BACKBONE}" SYNTH="$SYN" python ds_downstream.py
echo "=== $DS/$BACKBONE SUMMARY $(date) ==="; python ds_analyze.py "$OUT"
echo "=== DONE $(date) ==="
