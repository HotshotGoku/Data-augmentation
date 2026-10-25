#!/bin/bash
#SBATCH --job-name=faith_clean
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err

# P3 cheap-rank: faithfulness probe over ALL 8 clean configs {branching,mux,kl,rattray}x{clean,clean_dropout}.
# Ranks base vs dropout per dataset on its PRIMARY axis (branching->shape_factor, mux->atc, kl->ratio, rattray->strain).
# The probe trains on CLEAN reals, scores CLEAN synth zero-shot -> per-axis recovery (the cheap search objective).
#   DRYRUN=1  -> build+count only (no decode/train); run on CPU: sbatch -p common --export=ALL,DRYRUN=1 ...
#   normal    -> sbatch --export=ALL[,BACKBONE=resnet50][,SEEDS="0 1 2"] downstream/faithfulness_probe_clean_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
cd "$REPO"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export DRYRUN=${DRYRUN:-0}
[ "$DRYRUN" = "1" ] || python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

D=/hpc/group/youlab/sa603/data/clean
CSV=$REPO/downstream/outputs/branching_downstream_out
export OUT_TSV=$REPO/faithfulness_out/results_clean.tsv
export BACKBONE=${BACKBONE:-resnet18}
export EPOCHS=${EPOCHS:-30}
export NW=8
SEEDS=${SEEDS:-"0 1 2"}
[ "$DRYRUN" = "1" ] && SEEDS=0
rm -f "$OUT_TSV"

# CRITICAL: override builder defaults (which point at NON-clean dirs) with the clean reals.
export TRAIN_REAL_DIR=$D/branching_256/canon
export TEST_SRC_DIR=$D/branching_256/test/sources
export TEST_REAL_DIR=$D/branching_256/test/reals

for SEED in $SEEDS; do
  export SEED
  for V in clean clean_dropout; do
    echo "=== branching $V seed=$SEED $(date +%H:%M:%S) ==="
    TASK=branching FEATURES_CSV=$CSV/features_branching_$V.csv \
      SYNTH_DIR=$D/branching_downstream/synth_train_$V python downstream/faithfulness_probe.py || echo "  FAILED branching $V s$SEED"
    echo "=== mux $V seed=$SEED $(date +%H:%M:%S) ==="
    TASK=mux REALS_DIR=$D/multiplexed_eval/reals \
      SYNTH_DIR=$D/multiplexed_eval/synth_$V python downstream/faithfulness_probe.py || echo "  FAILED mux $V s$SEED"
    echo "=== kl $V seed=$SEED $(date +%H:%M:%S) ==="
    TASK=kl REALS_DIR=$D/ood_kl_2species \
      SYNTH_DIR=$D/ood_kl_2species/synth_$V python downstream/faithfulness_probe.py || echo "  FAILED kl $V s$SEED"
    echo "=== rattray $V seed=$SEED $(date +%H:%M:%S) ==="
    TASK=rattray REALS_DIR=$D/external_datasets/rattray_2023/processed \
      SYNTH_DIR=$D/external_datasets/rattray_2023/synth_$V python downstream/faithfulness_probe.py || echo "  FAILED rattray $V s$SEED"
  done
done
echo "=== PROBE DONE $(date) ==="
column -t -s$'\t' "$OUT_TSV" 2>/dev/null || cat "$OUT_TSV"
[ "$DRYRUN" = "1" ] && { echo "(dryrun: skip analyze)"; exit 0; }
echo "=== ANALYZE (rank base vs dropout per dataset on primary axis) ==="
python downstream/faithfulness_analyze.py "$OUT_TSV"
echo "=== DONE $(date) ==="
