#!/bin/bash
#SBATCH --job-name=sd15_divsweep
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Diversity-boosted SD1.5 sweep: trade conditioning strength / guidance for diversity, score each.
# Goal: find a config that raises diversity (gates downstream) without wrecking CMMD/realism.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export FT_EVAL_CKPT=$(ls -t finetune_runs/branching_sd15_fromseed/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -z "$FT_EVAL_CKPT" ] && { echo "FATAL: no sd15 branching ckpt"; exit 1; }
echo "ckpt=$FT_EVAL_CKPT"
REALS="FFBR:$D/branching_256/test/reals"

# (strength scale) configs: baseline + progressively less conditioning / guidance
for cfg in "1.0 9.0" "0.7 9.0" "0.5 9.0" "0.7 5.0" "0.5 5.0"; do
  set -- $cfg; S=$1; G=$2; TAG="s${S}_g${G}"
  OUT=$REPO/sd15_divsweep/$TAG
  echo "=== gen $TAG $(date) ==="
  GEN_OUT="$OUT" STRENGTH=$S SCALE=$G NUM_SAMPLES=4 python sd15_divgen.py || { echo "gen FAIL $TAG"; continue; }
  python eval_metrics.py --gen_dir "$OUT" --real_spec "$REALS" --out "$OUT/metrics" >/dev/null 2>&1 || echo "score FAIL $TAG"
done

echo "=== DIVERSITY-BOOSTED SD1.5 LEADERBOARD ==="
python - <<'PY'
import json, glob, os
print(f"{'config':14} {'realism':>8} {'CMMD':>7} {'diversity':>10}")
for m in sorted(glob.glob("/hpc/group/youlab/sa603/code/Data_augmentation/sd15_divsweep/*/metrics/metrics.json")):
    o=json.load(open(m)).get("overall",{}); g=lambda k:(o[k]["mean"] if isinstance(o.get(k),dict) else o.get(k))
    tag=os.path.basename(os.path.dirname(os.path.dirname(m)))
    try: print(f"{tag:14} {g('realism_vs_sibling'):8.3f} {g('CMMD'):7.2f} {g('diversity'):10.3f}")
    except: print(f"{tag:14} (parse err)")
print("baseline (strength 1.0): realism 0.324 / CMMD 3.38 / diversity 0.252 | rotation downstream is the bar to beat")
PY
echo "=== DONE $(date) ==="
