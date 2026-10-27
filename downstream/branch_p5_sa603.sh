#!/bin/bash
#SBATCH --job-name=branch_p5
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# P5 branching: 15-seed aug-vs-rotation on CLEAN data, resnet50 (standardized).
# Arms: real(floor) / real_dup(matched count, no diversity) / rotation / augmenter{clean,dropout,strat_sm}
#       + all-reps real ceiling. Scarcity N_train in {20,60}. All arms fully PAIRED (same seed subset).
# Headline per-axis = shape_factor R2; aggregates (shape-5, branch-arch-4) computed post-hoc by branch_p5_analyze.py.
# Resume-safe: re-submitting skips (tag,N_train,seed) rows already in the TSV.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data/clean
ODIR=$REPO/downstream/outputs/branching_downstream_out
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

# clean reals (shared across all arms) + per-config synth dirs + feature CSVs
export TRAIN_REAL_DIR=$D/branching_256/canon
export TEST_SRC_DIR=$D/branching_256/test/sources
export TEST_REAL_DIR=$D/branching_256/test/reals
CLEAN=$D/branching_downstream/synth_train_clean
DROPOUT=$D/branching_downstream/synth_train_clean_dropout
STRATSM=$D/branching_downstream/synth_train_clean_strat_sm
FCLEAN=$ODIR/features_branching_clean.csv            # reals + clean synth (reals shared across arms)
FDROPOUT=$ODIR/features_branching_clean_dropout.csv
FSTRATSM=$ODIR/features_branching_clean_strat_sm.csv

RESULTS=$ODIR/results_p5.tsv
mkdir -p "$ODIR"

done_already () { [ -f "$RESULTS" ] && awk -F'\t' -v t="$1" -v n="$2" -v s="$3" '$1==t&&$2==n&&$3==s{f=1} END{exit !f}' "$RESULTS"; }
run () { # tag mode synth csv nt seed
  if done_already "$1" "$5" "$6"; then echo "skip $1 N=$5 seed=$6 (done)"; return; fi
  echo "=== $1 N=$5 seed=$6 $(date +%H:%M:%S) ==="
  MODE="$2" MODE_TAG="$1" SYNTH_DIR="$3" FEATURES_CSV="$4" N_TRAIN="$5" SEED="$6" \
    BACKBONE=resnet50 EPOCHS=50 N_PER=8 NW=8 RESULTS_TSV="$RESULTS" \
    python downstream/downstream_colony_features.py || echo "  FAILED $1 N=$5 seed=$6"
}

for SEED in $(seq 0 14); do
  for NT in 20 60; do
    run real              real      "$CLEAN"   "$FCLEAN"   "$NT" "$SEED"
    run real_dup          real_dup  "$CLEAN"   "$FCLEAN"   "$NT" "$SEED"
    run rotation          classical "$CLEAN"   "$FCLEAN"   "$NT" "$SEED"
    run augmenter_clean   augmenter "$CLEAN"   "$FCLEAN"   "$NT" "$SEED"
    run augmenter_dropout augmenter "$DROPOUT" "$FDROPOUT" "$NT" "$SEED"
    run augmenter_stratsm augmenter "$STRATSM" "$FSTRATSM" "$NT" "$SEED"
  done
  run real_ceiling real "$CLEAN" "$FCLEAN" all "$SEED"   # upper-bound reference (all train conds)
done
echo "=== DONE $(date) ==="; column -t -s$'\t' "$RESULTS" 2>/dev/null | head -20
