#!/bin/bash
#SBATCH --job-name=ds_topoff
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Top-off the 7 rows the timed-out ResNet50 downstream run missed: aug_mplexft & aug_general at K=3
# seed 4, plus the real all-data ceiling (K=all, seeds 0-4). Appends to results_v3.tsv.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

RES=$REPO/downstream_multiplexed_out/results_v3.tsv
run () { env K=$3 MODE=$1 MODE_TAG=$2 SEED=$4 EPOCHS=80 BACKBONE=resnet50 RESULTS_TSV="$RES" ${5:+SYNTH_DIR=$5} \
  python downstream_multiplexed_decode.py || echo "  FAIL $2 K=$3 seed=$4"; }

echo "=== K=3 augmenter seed 4 (fills n=4 -> n=5) $(date) ==="
run augmenter aug_mplexft 3 4 "$MPX/synth_ft"
run augmenter aug_general 3 4 "$MPX/synth_generalist"
echo "=== real all-data ceiling, seeds 0-4 $(date) ==="
for SEED in 0 1 2 3 4; do run real real all "$SEED"; done
echo "=== DONE $(date) ==="; wc -l "$RES"
