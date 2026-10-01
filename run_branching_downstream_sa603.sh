#!/bin/bash
#SBATCH --job-name=branch_downstream
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Colony-feature downstream scarcity sweep: real vs classical(rotation) vs augmenter(synth),
# across N_train conditions and seeds. FEATURES_CSV must cover reals+synth (built by colony_features.py).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

export FEATURES_CSV=${FEATURES_CSV:-$REPO/branching_downstream_out/features_branching_all.csv}
export SYNTH_DIR=/hpc/group/youlab/sa603/data/branching_downstream/synth_train_generalist
export RESULTS_TSV=$REPO/branching_downstream_out/results_hardened.tsv
export BACKBONE=${BACKBONE:-resnet50}
export EPOCHS=${EPOCHS:-50}
export N_PER=${N_PER:-8}
rm -f "$RESULTS_TSV"

for NT in 20 60 all; do
  for MODE in real classical augmenter; do
    for SEED in 0 1 2 3 4; do
      echo "=== NT=$NT MODE=$MODE SEED=$SEED $(date +%H:%M:%S) ==="
      N_TRAIN=$NT MODE=$MODE SEED=$SEED python downstream_colony_features.py || echo "  FAILED NT=$NT MODE=$MODE SEED=$SEED"
    done
  done
done
echo "=== DONE $(date) ==="; column -t -s$'\t' "$RESULTS_TSV" 2>/dev/null || cat "$RESULTS_TSV"
