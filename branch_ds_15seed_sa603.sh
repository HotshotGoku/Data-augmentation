#!/bin/bash
#SBATCH --job-name=branch_ds_15seed
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Check #2: condition-subset sensitivity. Re-run the branching downstream for seeds 0-14 at the
# contested scarcity points (N_train=20,60) so the Flux-vs-rotation ranking is resolvable (n=5 isn't).
# All 4 arms share the identical per-seed condition subset (shuffle keys on SEED only) -> fully PAIRED.
# N=all omitted: already converged at 5 seeds. generalist arm omitted: dominated, not in the story.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data; ODIR=$REPO/branching_downstream_out
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

export BACKBONE=resnet50 EPOCHS=50 N_PER=8
RESULTS=$ODIR/results_15seed.tsv
rm -f "$RESULTS"
FLUX=$D/branching_downstream/synth_train_flux
BASE=$D/branching_downstream/synth_train_base
FCSV=$ODIR/features_branching_flux.csv     # canon+test reals + flux synth (reals shared across arms)
BCSV=$ODIR/features_branching_base.csv     # canon+test reals + base synth

run () { # mode_tag  mode  synth_dir  features_csv  N  seed
  echo "=== $1 N=$5 seed=$6 $(date +%H:%M:%S) ==="
  MODE="$2" MODE_TAG="$1" SYNTH_DIR="$3" FEATURES_CSV="$4" N_TRAIN="$5" SEED="$6" \
    RESULTS_TSV="$RESULTS" python downstream_colony_features.py || echo "  FAILED $1 N=$5 seed=$6"
}

for NT in 20 60; do
  for SEED in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    run real            real       "$FLUX" "$FCSV" "$NT" "$SEED"
    run classical       classical  "$FLUX" "$FCSV" "$NT" "$SEED"
    run augmenter_base  augmenter  "$BASE" "$BCSV" "$NT" "$SEED"
    run augmenter_flux  augmenter  "$FLUX" "$FCSV" "$NT" "$SEED"
  done
done
echo "=== DONE $(date) ==="; column -t -s$'\t' "$RESULTS" 2>/dev/null | head -20
