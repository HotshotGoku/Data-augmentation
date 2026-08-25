#!/bin/bash
#SBATCH --job-name=infer_sweep
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=6:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Inference hyperparameter sweep (scale x strength) on OUR checkpoint (CKPT_PATH_V4).
# Generates for OOD (held-out 2sp) + an in-domain guard, then scores each config with eval_metrics.py.
# Kinshuk: scale 9->15.1 + explore configs; we ALSO sweep control strength (the knob that targets the
# OOD "ignores the input" failure). Model is loaded ONCE by the driver.
#
#   sbatch Slurm_scripts/infer_sweep_sa603.sh
# Override grid:  sbatch --export=ALL,SWEEP_SCALES=9.0,15.1,SWEEP_STRENGTHS=1.0,2.0 Slurm_scripts/infer_sweep_sa603.sh

set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
TEST2SP=/hpc/group/youlab/sa603/data/finetune_2sp/test_2sp
INDOMAIN=/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3
OUTROOT=$REPO/infer_sweep_out
SCALES=${SWEEP_SCALES:-9.0,12.0,15.1}
STRENGTHS=${SWEEP_STRENGTHS:-1.0,1.5,2.0}
IND_MAXPREF=${IND_MAXPREF:-40}

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

mkdir -p "$OUTROOT" slurm_logs
echo "=== start $(date) | host $(hostname) | job ${SLURM_JOB_ID:-?} | scales=$SCALES strengths=$STRENGTHS ==="

echo "=== GENERATE (model loaded once) ==="
python reptorep_infer_sweep.py \
  --targets "OOD:${TEST2SP},IND:${INDOMAIN}:${IND_MAXPREF}" \
  --out_root "$OUTROOT" \
  --scales "$SCALES" --strengths "$STRENGTHS"
gen_status=$?
if [ $gen_status -ne 0 ]; then echo "FATAL: generation failed (exit $gen_status)"; exit $gen_status; fi

echo "=== SCORE each config ==="
SUMMARY="$OUTROOT/summary.tsv"
printf "target\tscale\tstrength\tFID\tdiversity\trealism_vs_sibling\tbaseline_real_vs_real\n" > "$SUMMARY"

score_one () {  # $1=gendir  $2=target  $3=real_spec
  local gendir="$1" target="$2" real_spec="$3" base scale strength
  base=$(basename "$gendir")                                   # e.g. scale9.0_str1.0
  scale=$(echo "$base"    | sed -E 's/scale([0-9.]+)_str.*/\1/')
  strength=$(echo "$base" | sed -E 's/.*_str([0-9.]+)/\1/')
  python eval_metrics.py --gen_dir "$gendir" --real_spec "$real_spec" --out "$gendir" || { echo "  eval FAILED $gendir"; return; }
  python - "$target" "$scale" "$strength" "$gendir/metrics.json" >> "$SUMMARY" <<'PY'
import json, sys
target, scale, strength, mj = sys.argv[1:5]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"{target}\t{scale}\t{strength}\t{o.get('FID')}\t{g('diversity')}\t{g('realism_vs_sibling')}\t{g('baseline_real_vs_real')}")
PY
}

for gendir in "$OUTROOT"/OOD/scale*_str*; do [ -d "$gendir" ] && score_one "$gendir" OOD   "2SP:${TEST2SP}";   done
for gendir in "$OUTROOT"/IND/scale*_str*; do [ -d "$gendir" ] && score_one "$gendir" IND   "FINAL:${INDOMAIN}"; done

echo "=== DONE $(date) — summary ==="
column -t -s $'\t' "$SUMMARY"
