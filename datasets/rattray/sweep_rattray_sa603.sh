#!/bin/bash
#SBATCH --job-name=rat_sweepeval
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Score the pre-finetune BASELINE + every Rattray fine-tune checkpoint on the held-out Rattray test.
# Decision metric: realism_vs_sibling vs baseline_real_vs_real (lower/closer = better) + diversity.
#   sbatch Slurm_scripts/sweep_rattray_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
RAT=/hpc/group/youlab/sa603/data/external_datasets/rattray_2023
TESTSRC=$RAT/test_sources
TESTREAL=$RAT/test_reals
OUTROOT=$REPO/rattray_sweep_eval
BASELINE=$REPO/lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.tsv"
printf "tag\tFID\tdiversity\trealism_vs_sibling\tbaseline_real_vs_real\n" > "$SUMMARY"

CKPTS=("$BASELINE")
while IFS= read -r c; do CKPTS+=("$c"); done < <(find "$REPO/rattray_runs" -name "*.ckpt" 2>/dev/null | sort)
echo "scoring ${#CKPTS[@]} checkpoints on $(ls "$TESTSRC"/*.TIF 2>/dev/null | wc -l) test-source strains"

for ckpt in "${CKPTS[@]}"; do
  if [ "$ckpt" = "$BASELINE" ]; then tag="baseline_base"
  else tag=$(echo "$ckpt" | sed -E 's#.*/rattray_runs/([^/]+)/.*/(epoch=[0-9]+)-.*#\1_\2#'); fi
  gen="$OUTROOT/$tag/gen"; res="$OUTROOT/$tag"
  echo "=== [$tag] ==="; mkdir -p "$gen"
  FT_EVAL_CKPT="$ckpt" python reptorep_infer_rattray.py --input_dir "$TESTSRC" --out_dir "$gen" || { echo "  infer FAILED"; continue; }
  python eval_metrics.py --gen_dir "$gen" --real_spec "RATTRAY:$TESTREAL" --out "$res" || { echo "  eval FAILED"; continue; }
  python - "$tag" "$res/metrics.json" >> "$SUMMARY" <<'PY'
import json, sys
tag, mj = sys.argv[1], sys.argv[2]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"{tag}\t{o.get('FID')}\t{g('diversity')}\t{g('realism_vs_sibling')}\t{g('baseline_real_vs_real')}")
PY
done
echo "=== DONE — summary ==="; column -t -s $'\t' "$SUMMARY"
