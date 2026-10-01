#!/bin/bash
#SBATCH --job-name=flux_score
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Standalone scoring of an existing Flux gen dir vs branching reals (clean GPU, no Flux in-process).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
GEN_OUT=${GEN_OUT:-$REPO/flux_cn_runs/branching_v2/gen}
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
cd "$REPO"
echo "=== score $GEN_OUT ($(ls "$GEN_OUT"/*.png 2>/dev/null | wc -l) imgs) vs branching reals $(date) ==="
python eval_metrics.py --gen_dir "$GEN_OUT" --real_spec "FFBR:/hpc/group/youlab/sa603/data/branching_256/test/reals" --out "$GEN_OUT/metrics"
python - "flux-branching-v2" "$GEN_OUT/metrics/metrics.json" <<'PY'
import json, sys
o = json.load(open(sys.argv[2])).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"RESULT {sys.argv[1]}: CMMD={g('CMMD')} realism_sib={g('realism_vs_sibling')} diversity={g('diversity')} copy_rate={g('copy_rate')}")
print("SD1.5 branching ref: realism 0.559 / CMMD 8.8 | PixCell-1024 ref: realism 0.599 / CMMD 55")
PY
echo "=== DONE $(date) ==="
