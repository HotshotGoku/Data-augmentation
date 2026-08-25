#!/bin/bash
#SBATCH --job-name=ft2sp_sweepeval
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Score the pre-fine-tune baseline + EVERY fine-tune checkpoint on the held-out 2sp test set.
# Per checkpoint: generate replicates on the test sources (reptorep_infer_ood.py, with the
# FT_EVAL_CKPT env override so pipeline loads THAT checkpoint), then compute metrics
# (eval_metrics.py). Collects one metrics.json per checkpoint + a summary.tsv.
#
# NOTE: the test set is small (~48 real / ~12 generated), so FID here is NOISY — the decision
# metric is realism_vs_sibling relative to baseline_real_vs_real (lower/closer = better), with
# diversity as a mode-collapse guard.
#
# Run AFTER at least one fine-tune has produced checkpoints:  sbatch Slurm_scripts/sweep_eval_2sp_sa603.sh

set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
TEST=/hpc/group/youlab/sa603/data/finetune_2sp/test_2sp
OUTROOT=$REPO/sweep_eval_2sp
BASELINE=$REPO/lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.tsv"
printf "tag\tFID\tdiversity\trealism_vs_sibling\tbaseline_real_vs_real\n" > "$SUMMARY"

# checkpoints to score: baseline first, then every fine-tune epoch checkpoint
CKPTS=("$BASELINE")
while IFS= read -r c; do CKPTS+=("$c"); done < <(find "$REPO/finetune_runs" -name "*.ckpt" 2>/dev/null | sort)
echo "scoring ${#CKPTS[@]} checkpoints on $(ls "$TEST"/*.TIF 2>/dev/null | wc -l) held-out test images"

for ckpt in "${CKPTS[@]}"; do
  if [ "$ckpt" = "$BASELINE" ]; then tag="baseline_fulldata"
  else tag=$(echo "$ckpt" | sed -E 's#.*/finetune_runs/([^/]+)/.*/(epoch=[0-9]+)-.*#\1_\2#'); fi
  gen="$OUTROOT/$tag/gen"; res="$OUTROOT/$tag"
  echo "=== [$tag] ==="; mkdir -p "$gen"
  FT_EVAL_CKPT="$ckpt" python reptorep_infer_ood.py --input_dir "$TEST" --out_dir "$gen" || { echo "  infer FAILED"; continue; }
  python eval_metrics.py --gen_dir "$gen" --real_spec "2SP:$TEST" --out "$res" || { echo "  eval FAILED"; continue; }
  python - "$tag" "$res/metrics.json" >> "$SUMMARY" <<'PY'
import json, sys
tag, mj = sys.argv[1], sys.argv[2]
o = json.load(open(mj))["overall"]
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"{tag}\t{o.get('FID')}\t{g('diversity')}\t{g('realism_vs_sibling')}\t{g('baseline_real_vs_real')}")
PY
done
echo "=== DONE — summary ==="; column -t -s $'\t' "$SUMMARY"
