#!/bin/bash
#SBATCH --job-name=dropout_confirm
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Confirm the headline positive result at 15 seeds: conditioning-dropout SD1.5 beats rotation on
# shape_factor. Synth + labels already exist -> just run downstream for seeds 5-14 (append to 0-4).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data; ODIR=$REPO/branching_downstream_out
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export FEATURES_CSV=$ODIR/features_branching_dropout05g9.csv
export SYNTH_DIR=$D/branching_downstream/synth_train_dropout05g9
export RESULTS_TSV=$ODIR/results_dropout05g9.tsv
export BACKBONE=resnet50 EPOCHS=50 N_PER=8 MODE=augmenter MODE_TAG=augmenter_dropout05g9

echo "=== dropout downstream seeds 5-14 (append) $(date) ==="
for NT in 20 60; do for SEED in 5 6 7 8 9 10 11 12 13 14; do
  echo "NT=$NT seed=$SEED $(date +%H:%M)"
  N_TRAIN=$NT SEED=$SEED python downstream_colony_features.py || echo "  FAIL NT=$NT seed=$SEED"
done; done

echo "=== 15-SEED PAIRED: conditioning-dropout vs rotation ==="
python3 -c "
import csv, statistics as st
def load(f): return list(csv.DictReader(open(f),delimiter='\t'))
drop={r['seed']:r for r in load('$ODIR/results_dropout05g9.tsv') if r['N_train']=='60'}
rot ={r['seed']:r for r in load('$ODIR/results_15seed.tsv') if r['mode']=='classical' and r['N_train']=='60'}
seeds=sorted(set(drop)&set(rot), key=int)
print(f'n={len(seeds)} paired seeds')
for metric in ['mean_r2','r2_shape_factor']:
    d=[float(drop[s][metric])-float(rot[s][metric]) for s in seeds]
    m=st.mean(d); sd=st.stdev(d); sem=sd/len(d)**0.5; t=m/sem
    print(f'  {metric:16} dropout-rotation {m:+.4f} +/-{sd:.4f} t={t:+.2f} {sum(x>0 for x in d)}/{len(d)} dropout>')
dm=st.mean([float(drop[s]['mean_r2']) for s in seeds]); rm=st.mean([float(rot[s]['mean_r2']) for s in seeds])
ds=st.mean([float(drop[s]['r2_shape_factor']) for s in seeds]); rs=st.mean([float(rot[s]['r2_shape_factor']) for s in seeds])
print(f'  means @N60: dropout meanR2 {dm:.3f} shape_factor {ds:.3f} | rotation meanR2 {rm:.3f} shape_factor {rs:.3f}')
"
echo "=== DONE $(date) ==="
