#!/bin/bash
#SBATCH --job-name=comp_reverify
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Re-verify with CLEAN DISHES: composite all branching data (reals + synth) to standardized dishes,
# re-run downstream (real/rotation/augmenter, all composited) at N=60, 15 seeds. Does the shape_factor
# win hold with clean dishes? Does standardization change aggregate R2?
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data; C=$D/branching_256; DS=$D/branching_downstream; ODIR=$REPO/branching_downstream_out
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home

echo "=== composite all data $(date) ==="
python dish_composite.py "$C/canon"        "$C/canon_comp"
python dish_composite.py "$C/test/sources" "$C/test/sources_comp"
python dish_composite.py "$C/test/reals"   "$C/test/reals_comp"
python dish_composite.py "$DS/synth_train_dropout_s10g7" "$DS/synth_train_dropout_s10g7_comp"

echo "=== label composited $(date) ==="
FCSV=$ODIR/features_branching_comp.csv
python colony_features.py --images "$C/canon_comp/*.TIF"        --out /tmp/cc_canon.csv
python colony_features.py --images "$C/test/sources_comp/*.TIF" --out /tmp/cc_tsrc.csv
python colony_features.py --images "$C/test/reals_comp/*.TIF"   --out /tmp/cc_treal.csv
python colony_features.py --images "$DS/synth_train_dropout_s10g7_comp/*.png" --out /tmp/cc_synth.csv
head -1 /tmp/cc_canon.csv > "$FCSV"; tail -q -n +2 /tmp/cc_canon.csv /tmp/cc_tsrc.csv /tmp/cc_treal.csv /tmp/cc_synth.csv >> "$FCSV"

echo "=== downstream on composited data (N=60, 15 seeds, arms real/classical/augmenter) $(date) ==="
export TRAIN_REAL_DIR=$C/canon_comp TEST_SRC_DIR=$C/test/sources_comp TEST_REAL_DIR=$C/test/reals_comp
export FEATURES_CSV="$FCSV" SYNTH_DIR=$DS/synth_train_dropout_s10g7_comp RESULTS_TSV=$ODIR/results_comp.tsv
export BACKBONE=resnet50 EPOCHS=50 N_PER=8
rm -f "$ODIR/results_comp.tsv"
for MODE in real classical augmenter; do for SEED in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  MODE=$MODE MODE_TAG=${MODE}_comp N_TRAIN=60 SEED=$SEED python downstream_colony_features.py || echo "  FAIL $MODE $SEED"
done; done

echo "=== CLEAN-DISH RESULT: augmenter vs rotation (composited, 15-seed, N=60) ==="
python3 -c "
import csv, statistics as st
r=list(csv.DictReader(open('$ODIR/results_comp.tsv'),delimiter='\t'))
def g(mode,metric): return {x['seed']:float(x[metric]) for x in r if x['mode']==mode}
for metric in ['mean_r2','r2_shape_factor']:
    a=g('augmenter_comp',metric); c=g('classical_comp',metric); s=sorted(set(a)&set(c),key=int)
    d=[a[k]-c[k] for k in s]; m=st.mean(d); sd=st.stdev(d); t=m/(sd/len(d)**0.5)
    print(f'  {metric:16} augmenter-rotation (clean dishes) {m:+.4f} t={t:+.2f} {sum(x>0 for x in d)}/{len(d)}>')
for mode in ['real_comp','classical_comp','augmenter_comp']:
    v=[float(x['mean_r2']) for x in r if x['mode']==mode]
    if v: print(f'  {mode:16} meanR2 {st.mean(v):.3f}  shape_factor {st.mean([float(x[\"r2_shape_factor\"]) for x in r if x[\"mode\"]==mode]):.3f}')
print('  prior (un-composited): dropout_s10g7 shape_factor +0.052 (t=2.14) over rotation; overall -0.037')
"
echo "=== DONE $(date) ==="
