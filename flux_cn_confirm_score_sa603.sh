#!/bin/bash
#SBATCH --job-name=flux_cnconf_score
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Score the full-scale cn0.7/cn0.6 (and re-score the cn1.0 baseline) apples-to-apples. Clean pytorch_PA job.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
cd "$REPO"
for d in flux_cn_runs/branching_v2/gen flux_cn_runs/cn07_full flux_cn_runs/cn06_full; do
  G=$REPO/$d
  python eval_metrics.py --gen_dir "$G" --real_spec "FFBR:/hpc/group/youlab/sa603/data/branching_256/test/reals" --out "$G/metrics" >/dev/null 2>&1 && \
  python - "$d" "$G/metrics/metrics.json" <<'PY' || echo "SCORE FAILED $1"
import json, sys
o = json.load(open(sys.argv[2])).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"CONFIRM {sys.argv[1]}: diversity={g('diversity'):.4f} realism={g('realism_vs_sibling'):.4f} CMMD={g('CMMD'):.3f} n_gen={g('copy_rate') is not None and 'ok'}")
PY
done
echo "=== SD1.5 branching ref: diversity ~0.266 realism 0.559 CMMD ~8.8-10.3 (generalist_eval_out ep5 ff) ==="
echo "=== DONE $(date) ==="
