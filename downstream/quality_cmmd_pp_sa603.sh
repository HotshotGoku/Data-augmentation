#!/bin/bash
#SBATCH --job-name=qual_pp
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err

# Synth-quality (FID + pooled CMMD) of the pp augmenter vs held-out reals, for the datasets with NO
# downstream label (pakp/nlev/selx/swarmevo). Realism-only: how faithfully does synth_pp track reals.
# Submit: sbatch --export=ALL,DS=<pakp|nlev|selx|swarmevo> downstream/quality_cmmd_pp_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh; conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$REPO/pipeline:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
DS=${DS:?set DS}
case "$DS" in
  pakp|nlev|selx) GEN=$D/${DS}_256_pp/synth_pp; REAL="$D/${DS}_256_pp/test/reals" ;;
  swarmevo)       GEN=$D/external_datasets/swarmevo/synth_pp     # reals nested per colony -> flatten via symlinks
                  REAL=$(mktemp -d); i=0
                  for f in "$D"/external_datasets/swarmevo/canon_pp/*/*.png; do ln -sf "$f" "$REAL/f$i.png"; i=$((i+1)); done
                  echo "swarmevo flat reals: $i frames" ;;
  *) echo "unknown DS=$DS"; exit 1 ;;
esac
OUT=$REPO/downstream/outputs/quality_pp/$DS; mkdir -p "$OUT"
echo "=== quality $DS $(date) gen=$GEN n_gen=$(ls "$GEN"/*.png 2>/dev/null | wc -l) ==="
python pipeline/metrics/eval_metrics.py --gen_dir "$GEN" --real_spec "${DS}:${REAL}" --out "$OUT"
echo "=== DONE $DS $(date) ==="
