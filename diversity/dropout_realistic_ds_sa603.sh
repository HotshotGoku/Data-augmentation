#!/bin/bash
#SBATCH --job-name=drop_real_ds
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Does the REALISTIC dropout config (full strength s1.0 / guidance 7) still beat rotation on shape_factor?
# Gen train-pool synth at s1.0g7, label, downstream 15 seeds (N=20,60), paired vs rotation.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data; ODIR=$REPO/branching_downstream_out
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
CKPT=$(ls -t finetune_runs/branching_sd15_dropout/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo FATAL no dropout ckpt; exit 1; }
TAG=dropout_s10g7; OUT=$D/branching_downstream/synth_train_$TAG; FCSV=$ODIR/features_branching_$TAG.csv

echo "=== gen realistic dropout synth (s1.0 g7) $(date) ==="
SYNTH_CKPT="$CKPT" SYNTH_DIR="$OUT" REAL_DIR=$D/branching_256/canon EXCLUDE_DIR=$D/branching_256/test/reals \
  STRENGTH=1.0 SCALE=7.0 SYNTH_PER_SRC=8 python gen_branching_synth.py
echo "count: $(ls "$OUT"/*.png 2>/dev/null | wc -l)"

echo "=== label ==="
python colony_features.py --images "$D/branching_256/canon/*.TIF"       --out /tmp/fr_canon.csv
python colony_features.py --images "$D/branching_256/test/sources/*.TIF" --out /tmp/fr_tsrc.csv
python colony_features.py --images "$D/branching_256/test/reals/*.TIF"    --out /tmp/fr_treal.csv
python colony_features.py --images "$OUT/*.png"                           --out /tmp/fr_synth.csv
head -1 /tmp/fr_canon.csv > "$FCSV"; tail -q -n +2 /tmp/fr_canon.csv /tmp/fr_tsrc.csv /tmp/fr_treal.csv /tmp/fr_synth.csv >> "$FCSV"

echo "=== downstream 15 seeds (N=20,60) ==="
export FEATURES_CSV="$FCSV" SYNTH_DIR="$OUT" RESULTS_TSV=$ODIR/results_$TAG.tsv
export BACKBONE=resnet50 EPOCHS=50 N_PER=8 MODE=augmenter MODE_TAG=augmenter_$TAG
rm -f "$ODIR/results_$TAG.tsv"
for NT in 20 60; do for SEED in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  N_TRAIN=$NT SEED=$SEED python downstream_colony_features.py || echo "  FAIL $NT $SEED"
done; done

echo "=== REALISTIC-CONFIG RESULT: paired vs rotation (15-seed) ==="
python3 -c "
import csv, statistics as st
def load(f): return list(csv.DictReader(open(f),delimiter='\t'))
dr={r['seed']:r for r in load('$ODIR/results_$TAG.tsv') if r['N_train']=='60'}
rot={r['seed']:r for r in load('$ODIR/results_15seed.tsv') if r['mode']=='classical' and r['N_train']=='60'}
seeds=sorted(set(dr)&set(rot), key=int); print(f'n={len(seeds)}')
for m in ['mean_r2','r2_shape_factor']:
    d=[float(dr[s][m])-float(rot[s][m]) for s in seeds]; mn=st.mean(d); sd=st.stdev(d); t=mn/(sd/len(d)**0.5)
    print(f'  {m:16} dropout_s10g7 - rotation {mn:+.4f} +/-{sd:.4f} t={t:+.2f} {sum(x>0 for x in d)}/{len(d)}>')
print('  (realistic config, diversity 0.271 vs the big/bright s0.5 config diversity 0.317)')
print('  prior s0.5g9 result: shape_factor +0.057 t=2.49 (won); overall tie')
"
echo "=== DONE $(date) ==="
