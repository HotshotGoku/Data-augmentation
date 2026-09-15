#!/bin/bash
#SBATCH --job-name=ds_mplex
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# DOWNSTREAM UTILITY: does augmenter-synth improve a multiplexed input-decoder in a data-scarce
# regime, beyond real-only and beyond classical rotation augmentation? Grid K{1,2,3} x
# MODE{real,classical,augmenter} x SEED{0,1,2} + real/all ceiling. Eval on held-out REAL test reps.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
SYNTH_DIR=$MPX/synth_ft
MPLEX_CKPT=$REPO/rattray_runs/multiplexed_ft/lightning_logs/version_54549115/checkpoints/epoch=2-step=1448.ckpt
OUT=$REPO/downstream_multiplexed_out

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
mkdir -p "$OUT" "$SYNTH_DIR"

# prefetch ImageNet weights (gpu nodes have internet); decoder random-inits if this fails
python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)" && echo "resnet weights cached" || echo "[warn] resnet prefetch failed"

# 1) generate synth once (idempotent: skip if already populated)
N_SYN=$(ls "$SYNTH_DIR"/*.png 2>/dev/null | wc -l)
if [ "$N_SYN" -lt 1000 ]; then
  echo "=== generating synth (have $N_SYN, want ~1680) $(date) ==="
  SYNTH_CKPT="$MPLEX_CKPT" SYNTH_DIR="$SYNTH_DIR" SYNTH_PER_SRC=8 python gen_multiplexed_synth.py
  echo "=== synth now: $(ls "$SYNTH_DIR"/*.png 2>/dev/null | wc -l) files $(date) ==="
else
  echo "=== synth exists ($N_SYN files), skipping gen ==="
fi

# 2) fresh results
rm -f "$OUT/results.tsv"

# 3) scarcity grid
for K in 1 2 3; do
  for MODE in real classical augmenter; do
    for SEED in 0 1 2; do
      echo "=== K=$K MODE=$MODE SEED=$SEED $(date) ==="
      K=$K MODE=$MODE SEED=$SEED N_PER=8 EPOCHS=40 SYNTH_DIR="$SYNTH_DIR" RESULTS_TSV="$OUT/results.tsv" \
        python downstream_multiplexed_decode.py || echo "  FAILED K=$K MODE=$MODE SEED=$SEED"
    done
  done
done
# ceiling: all train reps, real-only
for SEED in 0 1 2; do
  echo "=== K=all MODE=real SEED=$SEED $(date) ==="
  K=all MODE=real SEED=$SEED N_PER=8 EPOCHS=40 SYNTH_DIR="$SYNTH_DIR" RESULTS_TSV="$OUT/results.tsv" \
    python downstream_multiplexed_decode.py || echo "  FAILED all/real/$SEED"
done

echo "=== DONE $(date) — results.tsv ==="; column -t -s $'\t' "$OUT/results.tsv"
