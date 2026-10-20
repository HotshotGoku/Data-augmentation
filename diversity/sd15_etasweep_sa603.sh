#!/bin/bash
#SBATCH --job-name=sd15_etasweep
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Untried lever: eta (stochastic DDIM) — adds diversity WITHOUT lowering conditioning strength, so it
# may buy diversity without the realism/CMMD tax that strength/guidance reduction incurs.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export FT_EVAL_CKPT=$(ls -t finetune_runs/branching_sd15_fromseed/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -z "$FT_EVAL_CKPT" ] && { echo FATAL no ckpt; exit 1; }
REALS="FFBR:$D/branching_256/test/reals"

# (strength scale eta): keep conditioning full (strength 1.0) and inject stochasticity via eta; plus a couple combos
for cfg in "1.0 9.0 0.0" "1.0 9.0 0.5" "1.0 9.0 1.0" "1.0 5.0 1.0" "0.7 9.0 1.0"; do
  set -- $cfg; S=$1; G=$2; E=$3; TAG="s${S}_g${G}_e${E}"
  OUT=$REPO/sd15_etasweep/$TAG
  echo "=== gen $TAG $(date) ==="
  GEN_OUT="$OUT" STRENGTH=$S SCALE=$G ETA=$E NUM_SAMPLES=4 python sd15_divgen.py || { echo "gen FAIL $TAG"; continue; }
  python eval_metrics.py --gen_dir "$OUT" --real_spec "$REALS" --out "$OUT/metrics" >/dev/null 2>&1 || echo "score FAIL $TAG"
done

echo "=== ETA (STOCHASTIC) SWEEP LEADERBOARD ==="
python - <<'PY'
import json, glob, os
print(f"{'config(str_g_eta)':20} {'realism':>8} {'CMMD':>7} {'diversity':>10}")
for m in sorted(glob.glob("/hpc/group/youlab/sa603/code/Data_augmentation/sd15_etasweep/*/metrics/metrics.json")):
    o=json.load(open(m)).get("overall",{}); g=lambda k:(o[k]["mean"] if isinstance(o.get(k),dict) else o.get(k))
    tag=os.path.basename(os.path.dirname(os.path.dirname(m)))
    try: print(f"{tag:20} {g('realism_vs_sibling'):8.3f} {g('CMMD'):7.2f} {g('diversity'):10.3f}")
    except: print(f"{tag:20} (parse err)")
print("baseline s1.0_g9.0_e0.0: 0.324/3.38/0.252 | strength-sweep best-realistic s0.5_g9: 0.399/6.62/0.317")
print(">>> WIN = diversity up WITHOUT realism/CMMD climbing like the strength sweep did")
PY
echo "=== DONE $(date) ==="
