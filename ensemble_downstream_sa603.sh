#!/bin/bash
#SBATCH --job-name=sd15_ens_ds
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data; ODIR=$REPO/branching_downstream_out
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
CKPT=$(ls -t finetune_runs/branching_sd15_fromseed/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
TAG=ensemble; OUT=$D/branching_downstream/synth_train_$TAG; FCSV=$ODIR/features_branching_$TAG.csv

# Ensemble: 4 faithful (strength 1.0) + 4 diverse (strength 0.5) per source -> one synth set (new label info + coverage)
echo "=== gen ensemble (4 faithful + 4 diverse) $(date) ==="
SYNTH_CKPT="$CKPT" SYNTH_DIR="$OUT" REAL_DIR=$D/branching_256/canon EXCLUDE_DIR=$D/branching_256/test/reals \
  STRENGTH=1.0 SCALE=9.0 SYNTH_OFFSET=0 SYNTH_PER_SRC=4 python gen_branching_synth.py
SYNTH_CKPT="$CKPT" SYNTH_DIR="$OUT" REAL_DIR=$D/branching_256/canon EXCLUDE_DIR=$D/branching_256/test/reals \
  STRENGTH=0.5 SCALE=9.0 SYNTH_OFFSET=4 SYNTH_PER_SRC=4 python gen_branching_synth.py
echo "count: $(ls "$OUT"/*.png 2>/dev/null | wc -l)"

echo "=== label + downstream augmenter_$TAG ==="
python colony_features.py --images "$D/branching_256/canon/*.TIF"       --out /tmp/fe_canon.csv
python colony_features.py --images "$D/branching_256/test/sources/*.TIF" --out /tmp/fe_tsrc.csv
python colony_features.py --images "$D/branching_256/test/reals/*.TIF"    --out /tmp/fe_treal.csv
python colony_features.py --images "$OUT/*.png"                           --out /tmp/fe_synth.csv
head -1 /tmp/fe_canon.csv > "$FCSV"; tail -q -n +2 /tmp/fe_canon.csv /tmp/fe_tsrc.csv /tmp/fe_treal.csv /tmp/fe_synth.csv >> "$FCSV"
export FEATURES_CSV="$FCSV" SYNTH_DIR="$OUT" RESULTS_TSV=$ODIR/results_$TAG.tsv
export BACKBONE=resnet50 EPOCHS=50 N_PER=8 MODE=augmenter MODE_TAG=augmenter_$TAG
rm -f "$ODIR/results_$TAG.tsv"
for NT in 20 60 all; do for SEED in 0 1 2 3 4; do N_TRAIN=$NT SEED=$SEED python downstream_colony_features.py || echo "FAIL $NT $SEED"; done; done
python - <<PY
import csv, statistics as st
rows=list(csv.DictReader(open("$ODIR/results_$TAG.tsv"),delimiter="\t"))
for N in ["20","60","all"]:
    v=[float(r["mean_r2"]) for r in rows if r["N_train"]==N]
    if v: print(f"  N={N:3}: augmenter_ensemble meanR2 {st.mean(v):.3f}+/-{st.stdev(v):.3f}")
print("  REF @N60: rotation 0.443 | SD1.5-base 0.405 | Flux 0.431")
PY
echo "=== DONE $(date) ==="
