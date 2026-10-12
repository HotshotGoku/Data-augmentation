#!/bin/bash
#SBATCH --job-name=flux_scaled_score
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Score scaled-Flux generations vs test/reals with the SAME eval_metrics.py as SD1.5/pilot.
# Separate job from gen (pytorch_PA env; avoids the Flux-in-process eval_metrics segfault).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
G=${G:-$REPO/flux_cn_runs/scaled/eval_cn10}
echo "=== score $G $(date) ==="
python eval_metrics.py --gen_dir "$G" --real_spec "FFBR:$D/branching_256/test/reals" --out "$G/metrics"
python - "$G/metrics/metrics.json" <<'PY'
import json, sys
o = json.load(open(sys.argv[1])).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"SCALED Flux (CN4, cn1.0): diversity={g('diversity'):.4f} realism={g('realism_vs_sibling'):.4f} CMMD={g('CMMD'):.3f}")
print("  vs in-domain SD1.5:        0.252 / 0.324 / 3.38  (realism/CMMD to beat)")
print("  vs pilot Flux cn0.65+PG0.6: 0.270 / 0.524 / 8.78")
PY
echo "=== DONE $(date) ==="
