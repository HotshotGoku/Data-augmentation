#!/bin/bash
#SBATCH --job-name=sd15_div_ds
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# THE key test: does boosted-diversity SD1.5 beat rotation downstream? Generate a high-diversity synth
# TRAIN set (lower strength/guidance), auto-label, run the colony-feature downstream harness.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data; ODIR=$REPO/branching_downstream_out
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
CKPT=${CKPT:-$(ls -t finetune_runs/branching_sd15_fromseed/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)}
[ -z "$CKPT" ] && { echo FATAL no ckpt; exit 1; }
echo "using ckpt: $CKPT"
STRENGTH=${STRENGTH:-0.5}; SCALE=${SCALE:-9.0}; TAG=${TAG:-div05g9}
OUT=$D/branching_downstream/synth_train_$TAG; FCSV=$ODIR/features_branching_$TAG.csv

echo "=== gen boosted-diversity synth ($TAG str=$STRENGTH scale=$SCALE) $(date) ==="
SYNTH_CKPT="$CKPT" SYNTH_DIR="$OUT" REAL_DIR=$D/branching_256/canon EXCLUDE_DIR=$D/branching_256/test/reals \
  STRENGTH=$STRENGTH SCALE=$SCALE SYNTH_PER_SRC=8 python gen_branching_synth.py
echo "count: $(ls "$OUT"/*.png 2>/dev/null | wc -l)"

echo "=== label canon+test+synth -> $FCSV ==="
python colony_features.py --images "$D/branching_256/canon/*.TIF"       --out /tmp/fd_canon.csv
python colony_features.py --images "$D/branching_256/test/sources/*.TIF" --out /tmp/fd_tsrc.csv
python colony_features.py --images "$D/branching_256/test/reals/*.TIF"    --out /tmp/fd_treal.csv
python colony_features.py --images "$OUT/*.png"                           --out /tmp/fd_synth.csv
head -1 /tmp/fd_canon.csv > "$FCSV"
tail -q -n +2 /tmp/fd_canon.csv /tmp/fd_tsrc.csv /tmp/fd_treal.csv /tmp/fd_synth.csv >> "$FCSV"

echo "=== downstream augmenter_$TAG (N20/60/all x seeds0-4) ==="
export FEATURES_CSV="$FCSV" SYNTH_DIR="$OUT" RESULTS_TSV=$ODIR/results_$TAG.tsv
export BACKBONE=resnet50 EPOCHS=50 N_PER=8 MODE=augmenter MODE_TAG=augmenter_$TAG
rm -f "$ODIR/results_$TAG.tsv"
for NT in 20 60 all; do for SEED in 0 1 2 3 4; do
  N_TRAIN=$NT SEED=$SEED python downstream_colony_features.py || echo "  FAIL NT=$NT SEED=$SEED"
done; done

echo "=== RESULT: does boosted-diversity SD1.5 beat rotation? ==="
python - <<PY
import csv, statistics as st
rows=list(csv.DictReader(open("$ODIR/results_$TAG.tsv"),delimiter="\t"))
for N in ["20","60","all"]:
    v=[float(r["mean_r2"]) for r in rows if r["N_train"]==N]
    sf=[float(r["r2_shape_factor"]) for r in rows if r["N_train"]==N]
    if v: print(f"  N={N:3}: augmenter_$TAG meanR2 {st.mean(v):.3f}+/-{st.stdev(v):.3f}  shape_factor {st.mean(sf):.3f}")
print("  REF @N60 (15-seed): rotation 0.443 | SD1.5-base 0.405 | Flux 0.431 | real 0.410  <- BEAT ROTATION = the win")
PY
echo "=== DONE $(date) ==="
