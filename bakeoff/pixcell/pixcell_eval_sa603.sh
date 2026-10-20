#!/bin/bash
#SBATCH --job-name=pixcell_eval
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=5:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Score each PixCell-CN fine-tune epoch on the frozen multiplexed contract.
# Phase A: generate {prefix}_{n}.png per source (pixcell env). Phase B: eval_metrics v2 (pytorch_PA env).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
RUNS=$REPO/pixcell_cn_runs/mplex
OUTROOT=$REPO/pixcell_eval_out
EPOCHS="${EPOCHS_LIST:-5 10 15 20 25 30}"
mkdir -p "$OUTROOT"
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh

echo "############## PHASE A: generate (pixcell env) $(date) ##############"
conda activate /hpc/group/youlab/sa603/envs/pixcell
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export PYTHONPATH=/hpc/group/youlab/sa603/code/PixCell/controlnet:${PYTHONPATH:-}
cd "$REPO"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
for ep in $EPOCHS; do
  CKPT=$RUNS/controlnet_ep${ep}.pth
  [ -f "$CKPT" ] || { echo "skip ep$ep (no ckpt)"; continue; }
  echo "=== gen ep$ep $(date) ==="
  PIXCELL_CN_CKPT="$CKPT" GEN_OUT="$OUTROOT/pixcell_ep${ep}/gen" NUM_SAMPLES=2 STEPS=50 GUID=1.0 \
    python pixcell_gen_eval.py || echo "  gen FAILED ep$ep"
done
conda deactivate

echo "############## PHASE B: score (pytorch_PA env) $(date) ##############"
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
SUMMARY="$OUTROOT/summary.tsv"
printf "model\tCMMD\tFID\trealism_sib\trealism_src\tdiversity\tcopy_rate\n" > "$SUMMARY"
for ep in $EPOCHS; do
  GEN="$OUTROOT/pixcell_ep${ep}/gen"; RES="$OUTROOT/pixcell_ep${ep}"
  [ -d "$GEN" ] || { echo "skip ep$ep (no gen)"; continue; }
  echo "=== score ep$ep $(date) ==="
  python eval_metrics.py --gen_dir "$GEN" --real_spec "MPLEX:$MPX/reals_heldout" --out "$RES" || { echo "  eval FAILED ep$ep"; continue; }
  python - "pixcell_ep${ep}" "$RES/metrics.json" >> "$SUMMARY" <<'PY'
import json, sys
tag, mj = sys.argv[1], sys.argv[2]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"{tag}\t{g('CMMD')}\t{g('FID')}\t{g('realism_vs_sibling')}\t{g('realism_vs_source')}\t{g('diversity')}\t{g('copy_rate')}")
PY
done
echo "=== DONE $(date) — PixCell epoch curve (baseline SD1.5 mplex-ft = 0.4117 / 12.57 / div 0.381) ==="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k4 -g; } | column -t -s $'\t'
