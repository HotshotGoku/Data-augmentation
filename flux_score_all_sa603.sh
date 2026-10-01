#!/bin/bash
#SBATCH --job-name=flux_scoreall
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Score all diversity-sweep gen dirs in a CLEAN pytorch_PA job (no Flux in-process -> no segfault).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
cd "$REPO"
DIRS="divsweep/base divsweep/cn085 divsweep/cn07 divsweep/cn05 divsweep/g2 divsweep/g1 cadspg/nonebatch cadspg/cads05 cadspg/cads10 cadspg/cads20 cadspg/pg1 cadspg/pg4 cadspg/both"
echo "=== score all $(date) ==="
for d in $DIRS; do
  G=$REPO/flux_cn_runs/$d
  python eval_metrics.py --gen_dir "$G" --real_spec "FFBR:/hpc/group/youlab/sa603/data/branching_256/test/reals" --out "$G/metrics" >/dev/null 2>&1 && \
  python - "$d" "$G/metrics/metrics.json" <<'PY' || echo "SCORE FAILED $1"
import json, sys
o = json.load(open(sys.argv[2])).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"DIVRESULT {sys.argv[1]}: diversity={g('diversity'):.4f} realism={g('realism_vs_sibling'):.4f} CMMD={g('CMMD'):.3f}")
PY
done
echo "=== baseline (full v2 branching): diversity 0.220 realism 0.523 CMMD 6.52 | SD1.5 realism 0.559 ==="
echo "=== DONE $(date) ==="
