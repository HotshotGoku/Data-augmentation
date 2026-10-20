#!/bin/bash
#SBATCH --job-name=flux_pgfix_sc
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation; SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh; conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home; cd "$REPO"
for d in pgfix/pgf01 pgfix/pgf03 pgfix/pgf05 pgfix/pgf10; do
  G=$REPO/flux_cn_runs/$d
  python eval_metrics.py --gen_dir "$G" --real_spec "FFBR:/hpc/group/youlab/sa603/data/branching_256/test/reals" --out "$G/metrics" >/dev/null 2>&1 && \
  python - "$d" "$G/metrics/metrics.json" <<'PY' || echo "SCORE FAILED $1"
import json, sys
o = json.load(open(sys.argv[2])).get("overall", {}); g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"PGFIX {sys.argv[1]}: diversity={g('diversity'):.4f} realism={g('realism_vs_sibling'):.4f} CMMD={g('CMMD'):.3f}")
PY
done
echo "=== subset baseline cn1.0: div 0.247 real 0.511 CMMD 6.07 | SD1.5 div ~0.266 ==="; echo "=== DONE $(date) ==="
