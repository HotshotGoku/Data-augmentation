#!/bin/bash
#SBATCH --job-name=faith_probe
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err

# Faithfulness-probe VALIDATION: does the real-trained-probe recovery metric rank known win>lose?
#   mux synth_ft  -> expect row(aTc)/atc_r2 HIGH, col(IPTG)/iptg_r2 LOW (within-dataset aTc-win/IPTG-lose)
#   branching dropout_s10g7 (on-dist) vs base (washed, shape_factor bias +0.71) -> dropout shape_factor R2 > base
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
cd "$REPO"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

DATA=/hpc/group/youlab/sa603/data
CSV=$REPO/downstream/outputs/branching_downstream_out
export OUT_TSV=$REPO/faithfulness_out/results.tsv
export BACKBONE=${BACKBONE:-resnet18}
export EPOCHS=${EPOCHS:-30}
export NW=8
rm -f "$OUT_TSV"

for SEED in 0 1 2; do
  export SEED
  echo "=== mux synth_ft seed=$SEED $(date +%H:%M:%S) ==="
  TASK=mux SYNTH_DIR=$DATA/multiplexed_eval/synth_ft python downstream/faithfulness_probe.py || echo "  FAILED mux s$SEED"
  echo "=== branching dropout_s10g7 seed=$SEED $(date +%H:%M:%S) ==="
  TASK=branching FEATURES_CSV=$CSV/features_branching_dropout_s10g7.csv \
    SYNTH_DIR=$DATA/branching_downstream/synth_train_dropout_s10g7 python downstream/faithfulness_probe.py || echo "  FAILED dropout s$SEED"
  echo "=== branching base seed=$SEED $(date +%H:%M:%S) ==="
  TASK=branching FEATURES_CSV=$CSV/features_branching_base.csv \
    SYNTH_DIR=$DATA/branching_downstream/synth_train_base python downstream/faithfulness_probe.py || echo "  FAILED base s$SEED"
done
echo "=== DONE $(date) ==="
column -t -s$'\t' "$OUT_TSV" 2>/dev/null || cat "$OUT_TSV"
