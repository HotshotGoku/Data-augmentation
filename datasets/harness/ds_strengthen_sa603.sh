#!/bin/bash
#SBATCH --job-name=ds_strengthen
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Airtight-negative check: gen from the LONGER-trained in-domain ckpt -> 15-seed downstream vs rotation.
# If a stronger augmenter still doesn't beat rotation, the negative is robust. sbatch --export=ALL,DS=<kl|rattray>
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
case "${DS:?set DS}" in
  rattray) SRCG="$D/external_datasets/rattray_2023/processed/*.png"; REALS_G="$SRCG"; LRE='PIL(?P<lab>[0-9]+)-(?P<rep>[0-9]+)'; KI=cls; LG=0 ;;
  kl)      SRCG="$D/ood_kl_2species/*.TIF"; REALS_G="$SRCG"; LRE='(?P<lab>[0-9]+-[0-9]+)_(?P<rep>[0-9]+)\.TIF'; KI=reg; LG=1 ;;
  *) echo "unknown DS"; exit 1 ;;
esac
CK=$(ls -t "$REPO"/finetune_runs/${DS}_sd15_long/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
[ -f "$CK" ] || { echo "FATAL no long ckpt"; exit 1; }
echo "=== GEN from longer ckpt $CK $(date) ==="
FT_EVAL_CKPT="$CK" SOURCES="$SRCG" OUT="$REPO/${DS}_synth_long" N=8 STRENGTH=${STR:-1.0} SCALE=${SCL:-9.0} python ds_gen.py

echo "=== DOWNSTREAM 15-seed (real/rotation/aug-long) $(date) ==="
export REALS="$REALS_G" LABEL_RE="$LRE" KIND="$KI" LOGRATIO="$LG" TEST_REP=1 K=1 N_PER=8 SEEDS=15 OUT="$REPO/${DS}_downstream_long.tsv"
rm -f "$OUT"
MODE=real      TAG="$DS"       python ds_downstream.py
MODE=rotation  TAG="$DS"       python ds_downstream.py
MODE=augmenter TAG="${DS}_long" SYNTH="$REPO/${DS}_synth_long" python ds_downstream.py
echo "=== $DS STRENGTHEN SUMMARY $(date) ==="; python ds_analyze.py "$OUT"
echo "=== DONE $(date) ==="
