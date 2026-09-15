#!/bin/bash
#SBATCH --job-name=ds_mplex_v2
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# WS1b: strengthen the downstream aTc/IPTG decoder. Adds a generalist-augmenter arm and 5 seeds
# (error bars). Arms: real | classical | aug_mplexft (dedicated synth) | aug_general (generalist synth).
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
python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)" >/dev/null 2>&1 || true

GEN_CKPT=$REPO/rattray_runs/generalist/lightning_logs/version_54592625/checkpoints/epoch=3-step=5723.ckpt
SYNTH_FT=$MPX/synth_ft
SYNTH_GEN=$MPX/synth_generalist
RES=$REPO/downstream_multiplexed_out/results_v2.tsv

# generate generalist-augmenter synth once (mirrors the dedicated synth_ft: reps 8,9,10 x8)
if [ "$(ls "$SYNTH_GEN"/*.png 2>/dev/null | wc -l)" -lt 1000 ]; then
  echo "=== generating generalist synth $(date) ==="
  SYNTH_CKPT="$GEN_CKPT" SYNTH_DIR="$SYNTH_GEN" SYNTH_PER_SRC=8 python gen_multiplexed_synth.py
fi

ARMS=${ARMS:-all}; [ "$ARMS" = "all" ] && rm -f "$RES"
run () {  # $1=MODE $2=MODE_TAG $3=K $4=SEED [$5=SYNTH_DIR]  (env parses runtime VAR=val prefixes)
  env K=$3 MODE=$1 MODE_TAG=$2 SEED=$4 EPOCHS=40 RESULTS_TSV="$RES" ${5:+SYNTH_DIR=$5} \
    python downstream_multiplexed_decode.py || echo "  FAIL $2 K=$3 seed=$4"
}
for K in 1 2 3; do
  for SEED in 0 1 2 3 4; do
    echo "=== K=$K SEED=$SEED $(date) ==="
    if [ "$ARMS" = "all" ]; then
      run real       real        "$K" "$SEED"
      run classical  classical   "$K" "$SEED"
    fi
    run augmenter  aug_mplexft "$K" "$SEED" "$SYNTH_FT"
    run augmenter  aug_general "$K" "$SEED" "$SYNTH_GEN"
  done
done
[ "$ARMS" = "all" ] && for SEED in 0 1 2 3 4; do run real real all "$SEED"; done

echo "=== DONE $(date) — results_v2.tsv ==="; column -t -s $'\t' "$RES" | head -60
