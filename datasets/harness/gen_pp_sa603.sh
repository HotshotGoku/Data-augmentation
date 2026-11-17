#!/bin/bash
#SBATCH --job-name=gen_pp
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Gen synth from a <ds>_sd15_pp (new per-dataset preprocess + runtime-aug) ckpt, into the pp-domain
# synth dir -- so synth, reals and the ckpt all share ONE preprocessing domain (no domain-shift confound).
#   metric datasets (branching/mux/kl/rattray): synth of the TRAIN pool -> downstream augmentation.
#   quality datasets (pakp/nlev/selx/swarmevo): synth of held-out TEST sources -> CMMD vs test reals.
# Adapted from gen_clean_sa603.sh (old clean domain, mux/kl/rattray only) for the pp domain + all 8.
# Submit: sbatch --export=ALL,DS=<ds>,SYNTH_CKPT=<ckpt>[,SYNTH_PER_SRC=8] datasets/harness/gen_pp_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$REPO/pipeline:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"  # pipeline/ for import pipeline; utils symlink for Data_augmentation.utils
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

DS=${DS:?set DS}; CK=${SYNTH_CKPT:?set SYNTH_CKPT}; PER=${SYNTH_PER_SRC:-8}
echo "=== gen_pp DS=$DS $(date) ckpt=$CK PER=$PER ==="
case "$DS" in
  branching)  # train-pool synth (excludes the held-out test conditions) via gen_branching_synth.py
     OUT=$D/branching_256_pp/synth_train_pp
     SYNTH_CKPT="$CK" REAL_DIR=$D/branching_256_pp/canon EXCLUDE_DIR=$D/branching_256_pp/test/reals \
       SYNTH_DIR="$OUT" SYNTH_PER_SRC="$PER" python downstream/gen_branching_synth.py ;;
  mux)        # gen_multiplexed_synth.py loops train reps 8,9,10 from multiplexed_eval/reals (as-is fluor)
     OUT=$D/multiplexed_eval/synth_pp
     SYNTH_CKPT="$CK" SYNTH_DIR="$OUT" SYNTH_PER_SRC="$PER" python datasets/multiplexed/gen_multiplexed_synth.py ;;
  kl)         OUT=$D/kl_256_pp/synth_pp; mkdir -p "$OUT"
     FT_EVAL_CKPT="$CK" SOURCES="$D/kl_256_pp/canon/*.TIF" OUT="$OUT" N="$PER" python datasets/harness/ds_gen.py ;;
  rattray)    OUT=$D/external_datasets/rattray_2023/synth_pp; mkdir -p "$OUT"
     FT_EVAL_CKPT="$CK" SOURCES="$D/external_datasets/rattray_2023/processed_pp/*.png" OUT="$OUT" N="$PER" python datasets/harness/ds_gen.py ;;
  pakp|nlev|selx) OUT=$D/${DS}_256_pp/synth_pp; mkdir -p "$OUT"   # quality: gen over held-out test sources
     FT_EVAL_CKPT="$CK" SOURCES="$D/${DS}_256_pp/test/sources/*.TIF" OUT="$OUT" N="$PER" python datasets/harness/ds_gen.py ;;
  swarmevo)   OUT=$D/external_datasets/swarmevo/synth_pp; mkdir -p "$OUT"  # quality: one source frame per colony
     FT_EVAL_CKPT="$CK" SOURCES="$D/external_datasets/swarmevo/canon_pp/*/00000.png" OUT="$OUT" N="$PER" python datasets/harness/ds_gen.py ;;
  *) echo "unknown DS=$DS"; exit 1 ;;
esac
echo "=== gen_pp done $(date) count=$(ls "$OUT"/*.png 2>/dev/null | wc -l) -> $OUT ==="
