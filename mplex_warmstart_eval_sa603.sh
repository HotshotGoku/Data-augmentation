#!/bin/bash
#SBATCH --job-name=mplex_ws_eval
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Warm-start A/B: does specializing on multiplexed FROM the generalist beat FROM base (dedicated 1.52x)?
# Scores base + dedicated(from-base, ep2) + every warm-start(from-generalist) epoch vs held-out val+test.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
OUTROOT=$REPO/mplex_warmstart_eval_out
BASE=$REPO/lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt
DEDICATED=$REPO/rattray_runs/multiplexed_ft/lightning_logs/version_54549115/checkpoints/epoch=2-step=1448.ckpt

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.tsv"
printf "model\tFID\tdiversity\trealism_vs_sibling\tbaseline_real_vs_real\n" > "$SUMMARY"

score () {  # $1=tag $2=ckpt
  local tag="$1" ckpt="$2" gen="$OUTROOT/$1/gen" res="$OUTROOT/$1"
  echo "=== [$tag] ==="; rm -rf "$res"; mkdir -p "$gen"
  FT_EVAL_CKPT="$ckpt" python reptorep_infer_rattray.py --input_dir "$MPX/sources" --out_dir "$gen" || { echo "  infer FAILED"; return; }
  python eval_metrics.py --gen_dir "$gen" --real_spec "MPLEX:$MPX/reals_heldout" --out "$res" || { echo "  eval FAILED"; return; }
  python - "$tag" "$res/metrics.json" >> "$SUMMARY" <<'PY'
import json, sys
tag, mj = sys.argv[1], sys.argv[2]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"{tag}\t{o.get('FID')}\t{g('diversity')}\t{g('realism_vs_sibling')}\t{g('baseline_real_vs_real')}")
PY
}

score "base"          "$BASE"
score "dedicated_ep2" "$DEDICATED"
while IFS= read -r c; do
  ep=$(echo "$c" | grep -oE 'epoch=[0-9]+')
  score "warmstart_$ep" "$c"
done < <(find "$REPO/rattray_runs/mplex_warmstart" -name "*.ckpt" 2>/dev/null | sort)

echo "=== DONE $(date) — multiplexed realism (lower=better; dedicated from-base ep2 = 0.412/1.52x to beat) ==="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k4 -g; } | column -t -s $'\t'
