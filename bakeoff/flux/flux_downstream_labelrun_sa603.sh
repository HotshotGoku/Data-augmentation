#!/bin/bash
#SBATCH --job-name=flux_ds_run
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Label reals+Flux-synth, then run the downstream augmenter_flux arm (real/classical/base reused from
# results_hardened.tsv). Same harness/split as the hardened run -> directly comparable.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation; SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data; ODIR=$REPO/branching_downstream_out
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh; conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
cd "$REPO"
FLUXSYNTH=$D/branching_downstream/synth_train_flux

echo "=== LABEL canon + test + flux-synth -> features_branching_flux.csv $(date) ==="
python colony_features.py --images "$D/branching_256/canon/*.TIF"       --out /tmp/ff_canon.csv || echo "canon FAIL"
python colony_features.py --images "$D/branching_256/test/sources/*.TIF" --out /tmp/ff_tsrc.csv || echo "tsrc FAIL"
python colony_features.py --images "$D/branching_256/test/reals/*.TIF"    --out /tmp/ff_treal.csv || echo "treal FAIL"
python colony_features.py --images "$FLUXSYNTH/*.png"                     --out /tmp/ff_synth.csv || echo "synth FAIL"
head -1 /tmp/ff_canon.csv > "$ODIR/features_branching_flux.csv"
tail -q -n +2 /tmp/ff_canon.csv /tmp/ff_tsrc.csv /tmp/ff_treal.csv /tmp/ff_synth.csv >> "$ODIR/features_branching_flux.csv"
echo "=== labeled $(($(wc -l < "$ODIR/features_branching_flux.csv")-1)) images ==="

echo "=== RUN downstream augmenter_flux arm $(date) ==="
export FEATURES_CSV=$ODIR/features_branching_flux.csv SYNTH_DIR=$FLUXSYNTH RESULTS_TSV=$ODIR/results_flux.tsv
export BACKBONE=resnet50 EPOCHS=50 N_PER=8 MODE=augmenter MODE_TAG=augmenter_flux
rm -f "$RESULTS_TSV"
for NT in 20 60 all; do for SEED in 0 1 2 3 4; do
  echo "=== NT=$NT augmenter_flux SEED=$SEED $(date +%H:%M:%S) ==="
  N_TRAIN=$NT SEED=$SEED python downstream_colony_features.py || echo "  FAILED NT=$NT SEED=$SEED"
done; done
echo "=== DONE $(date) ==="; column -t -s$'\t' "$RESULTS_TSV" 2>/dev/null | head
