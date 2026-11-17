#!/bin/bash
#SBATCH --job-name=feats_branch_pp
#SBATCH -p common
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err

# Label the pp-domain branching reals (canon + test) + synth_train_pp with colony_features
# (shape-5 + branch-arch-4), building features_branching_pp.csv that downstream_pp_sa603.sh reads.
# Adapted from rebuild_branch_features_sa603.sh (clean domain, 3 synth configs) for the pp domain.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
D=/hpc/group/youlab/sa603/data/branching_256_pp
ODIR=$REPO/downstream/outputs/branching_downstream_out
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh; conda activate pytorch_PA_patternprediction
cd "$REPO"; mkdir -p "$ODIR"
CF="python downstream/colony_features.py"
T=$(mktemp -d)

echo "=== selfcheck ==="; $CF --selfcheck || { echo "SELFCHECK FAILED - aborting"; exit 1; }

echo "=== label reals (canon + test/sources + test/reals) ==="
$CF --images "$D/canon/*.TIF"            --out "$T/r1.csv"
$CF --images "$D/test/sources/*_src.TIF" --out "$T/r2.csv"
$CF --images "$D/test/reals/*_Rep*.TIF"  --out "$T/r3.csv"
head -1 "$T/r1.csv" > "$ODIR/features_branching_pp.csv"
for f in r1 r2 r3; do tail -n+2 "$T/$f.csv" >> "$ODIR/features_branching_pp.csv"; done
echo "reals rows: $(($(wc -l < "$ODIR/features_branching_pp.csv")-1))"

echo "=== label synth_train_pp ==="
$CF --images "$D/synth_train_pp/*.png" --out "$T/s.csv"
tail -n+2 "$T/s.csv" >> "$ODIR/features_branching_pp.csv"
echo "features_branching_pp.csv total rows: $(($(wc -l < "$ODIR/features_branching_pp.csv")-1))"
echo "=== DONE $(date) ==="
