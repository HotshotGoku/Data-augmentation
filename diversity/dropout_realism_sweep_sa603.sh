#!/bin/bash
#SBATCH --job-name=drop_realsweep
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Find the realism sweet spot of the CONDITIONING-DROPOUT model: sweep higher strengths/guidance
# (more source-anchoring = more realistic) and score realism/CMMD/diversity. Goal: realistic AND still diverse.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export FT_EVAL_CKPT=$(ls -t finetune_runs/branching_sd15_dropout/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -z "$FT_EVAL_CKPT" ] && { echo FATAL no dropout ckpt; exit 1; }
echo "dropout ckpt=$FT_EVAL_CKPT"
REALS="FFBR:$D/branching_256/test/reals"

# higher strength = more faithful (realistic) ; include the old s0.5 for reference
for cfg in "1.0 9.0" "0.85 9.0" "0.7 9.0" "1.0 7.0" "0.5 9.0"; do
  set -- $cfg; S=$1; G=$2; TAG="drop_s${S}_g${G}"
  OUT=$REPO/dropout_realsweep/$TAG
  echo "=== gen $TAG $(date) ==="
  GEN_OUT="$OUT" STRENGTH=$S SCALE=$G NUM_SAMPLES=4 python sd15_divgen.py || { echo "gen FAIL $TAG"; continue; }
  python eval_metrics.py --gen_dir "$OUT" --real_spec "$REALS" --out "$OUT/metrics" >/dev/null 2>&1 || echo "score FAIL $TAG"
done

echo "=== DROPOUT-MODEL REALISM SWEEP (want: realism near baseline 0.324 AND diversity > 0.27) ==="
python - <<'PY'
import json, glob, os
print(f"{'config':18} {'realism':>8} {'CMMD':>7} {'diversity':>10}")
for m in sorted(glob.glob("/hpc/group/youlab/sa603/code/Data_augmentation/dropout_realsweep/*/metrics/metrics.json")):
    o=json.load(open(m)).get("overall",{}); g=lambda k:(o[k]["mean"] if isinstance(o.get(k),dict) else o.get(k))
    tag=os.path.basename(os.path.dirname(os.path.dirname(m)))
    try: print(f"{tag:18} {g('realism_vs_sibling'):8.3f} {g('CMMD'):7.2f} {g('diversity'):10.3f}")
    except: print(f"{tag:18} (parse err)")
print("REF: baseline SD1.5 s1.0 = realism 0.324 / CMMD 3.38 / div 0.252 (most realistic) | shown-in-montage s0.5 = 0.399/6.62/0.317 (too big/bright)")
print("real-vs-real floor realism ~0.25")
PY
echo "=== DONE $(date) ==="
