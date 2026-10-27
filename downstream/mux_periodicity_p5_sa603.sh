#!/bin/bash
#SBATCH --job-name=mux_period_p5
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# P5 mux PERIODICITY (perim_roughness/n_lobes, the NEW aTc-linked morphology axis): 15-seed aug-vs-rotation
# on CLEAN data, resnet50. Arms: real/real_dup/rotation/augmenter{base,dropout} + all-reps ceiling. K in {1,2,3}.
# Each synth carries its OWN recomputed periodicity (label diversity); rotation copies keep the SAME label.
# Resume-safe: skips (tag,K,seed) rows already in the TSV.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data/clean
ODIR=$REPO/downstream/outputs/mux_periodicity_out
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

REALS=$D/multiplexed_eval/reals
BASE=$D/multiplexed_eval/synth_clean
DROPOUT=$D/multiplexed_eval/synth_clean_dropout
CSV=$ODIR/periodicity_clean_all.csv                  # reals + base+dropout synth periodicity labels
RESULTS=$ODIR/results_p5.tsv
mkdir -p "$ODIR"

done_already () { [ -f "$RESULTS" ] && awk -F'\t' -v t="$1" -v k="$2" -v s="$3" '$1==t&&$2==k&&$3==s{f=1} END{exit !f}' "$RESULTS"; }
run () { # tag mode synth k seed
  if done_already "$1" "$4" "$5"; then echo "skip $1 K=$4 seed=$5 (done)"; return; fi
  echo "=== $1 K=$4 seed=$5 $(date +%H:%M:%S) ==="
  MODE="$2" MODE_TAG="$1" SYNTH_DIR="$3" K="$4" SEED="$5" \
    BACKBONE=resnet50 EPOCHS=40 N_PER=8 NW=8 REALS_DIR="$REALS" MUX_PERIODICITY_CSV="$CSV" RESULTS_TSV="$RESULTS" \
    python downstream/downstream_mux_periodicity.py || echo "  FAILED $1 K=$4 seed=$5"
}

for SEED in $(seq 0 14); do
  for K in 1 2 3; do
    run real              real      "$BASE"    "$K" "$SEED"
    run real_dup          real_dup  "$BASE"    "$K" "$SEED"
    run rotation          classical "$BASE"    "$K" "$SEED"
    run augmenter_base    augmenter "$BASE"    "$K" "$SEED"
    run augmenter_dropout augmenter "$DROPOUT" "$K" "$SEED"
  done
  run real_ceiling real "$BASE" all "$SEED"   # upper-bound reference (all train reps)
done
echo "=== DONE $(date) ==="; column -t -s$'\t' "$RESULTS" 2>/dev/null | head -20
