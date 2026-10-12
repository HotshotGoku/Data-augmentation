#!/bin/bash
#SBATCH --job-name=mux_k1_15seed
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Rigorous generalization confirm: multiplexed DECODE, K=1 (scarcest), 15 seeds, resnet50.
# arms real/classical/aug_mplexft -> does the augmenter beat rotation on aTc decode (row_acc) at 15 seeds?
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
ODIR=$REPO/downstream_multiplexed_out
export RESULTS_TSV=$ODIR/results_mux15_k1.tsv BACKBONE=resnet50 EPOCHS=40 N_PER=8 K=1
rm -f "$RESULTS_TSV"
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  MODE=real       MODE_TAG=real       SEED=$SEED python downstream_multiplexed_decode.py || echo "FAIL real $SEED"
  MODE=classical  MODE_TAG=classical  SEED=$SEED python downstream_multiplexed_decode.py || echo "FAIL classical $SEED"
  MODE=augmenter  MODE_TAG=aug_mplexft SYNTH_DIR=/hpc/group/youlab/sa603/data/multiplexed_eval/synth_ft SEED=$SEED python downstream_multiplexed_decode.py || echo "FAIL aug $SEED"
done

echo "=== MULTIPLEXED 15-seed paired: aug_mplexft vs rotation (K=1) ==="
python3 -c "
import csv, statistics as st
r=list(csv.DictReader(open('$RESULTS_TSV'),delimiter='\t'))
def g(m,metric): return {x['seed']:float(x[metric]) for x in r if x['mode']==m}
for metric in ['joint_acc','row_acc','col_acc','atc_r2','iptg_r2']:
    a=g('aug_mplexft',metric); c=g('classical',metric); s=sorted(set(a)&set(c),key=int)
    if not s: continue
    d=[a[k]-c[k] for k in s]; m=st.mean(d); sd=st.stdev(d); t=m/(sd/len(d)**0.5) if sd else 0
    print(f'  {metric:10} aug {st.mean([a[k] for k in s]):.3f} vs rot {st.mean([c[k] for k in s]):.3f}  d={m:+.4f} t={t:+.2f} {sum(x>0 for x in d)}/{len(d)}>')
print('  (5-seed had: row_acc +0.055 t=6.98; joint_acc +0.010 t=2.58)')
"
echo "=== DONE $(date) ==="
