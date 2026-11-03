#!/bin/bash
#SBATCH --job-name=rebuild_branch_feats
#SBATCH -p common
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err

# Re-label branching reals + the 3 synth sets with the FIXED colony_features (prune+dedup junctions),
# rebuilding features_branching_{clean,clean_dropout,clean_strat_sm}.csv. CPU. Aborts if selfcheck fails.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
D=/hpc/group/youlab/sa603/data/clean/branching_256
SD=/hpc/group/youlab/sa603/data/clean/branching_downstream
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
head -1 "$T/r1.csv" > "$T/reals.csv"
for f in r1 r2 r3; do tail -n+2 "$T/$f.csv" >> "$T/reals.csv"; done
echo "reals rows: $(($(wc -l < "$T/reals.csv")-1))"

for cfg in clean clean_dropout clean_strat_sm; do
  echo "=== label synth_train_$cfg ==="
  $CF --images "$SD/synth_train_$cfg/*.png" --out "$T/s_$cfg.csv"
  cp "$T/reals.csv" "$ODIR/features_branching_$cfg.csv"
  tail -n+2 "$T/s_$cfg.csv" >> "$ODIR/features_branching_$cfg.csv"
  echo "features_branching_$cfg.csv rows: $(($(wc -l < "$ODIR/features_branching_$cfg.csv")-1))"
done
echo "=== corrected n_branch_points summary reals vs synth ==="
python3 -c "
import csv,statistics as st
r=list(csv.DictReader(open('$ODIR/features_branching_clean.csv')))
real=[float(x['n_branch_points']) for x in r if x['name'].endswith('.TIF')]
syn=[float(x['n_branch_points']) for x in r if x['name'].endswith('.png')]
print('reals  n_branch_points mean %.1f median %.0f max %.0f'%(st.mean(real),st.median(real),max(real)))
print('synth  n_branch_points mean %.1f median %.0f max %.0f'%(st.mean(syn),st.median(syn),max(syn)))
"
echo "=== DONE $(date) ==="
