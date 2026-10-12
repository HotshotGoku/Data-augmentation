#!/bin/bash
#SBATCH --job-name=sd15_branch_eval
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Check #1 part 2: gen 4 samples/source from the retrained in-domain SD1.5 on the 28 held-out test
# sources, then score realism/CMMD/diversity vs test/reals with the SAME eval_metrics.py as Flux.
# gen and score are separate python processes (pipeline.py's deterministic state can't leak into eval).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home

CKPT=${CKPT:-$(ls -t finetune_runs/branching_sd15_fromseed/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)}
[ -z "${CKPT:-}" ] && { echo "FATAL: no branching SD1.5 checkpoint found"; exit 1; }
export FT_EVAL_CKPT="$CKPT"
GEN_OUT=$REPO/sd15_branching_eval
echo "=== SD1.5 in-domain gen+score  ckpt=$CKPT  $(date) ==="

GEN_OUT="$GEN_OUT" SOURCES="$D/branching_256/test/sources" NUM_SAMPLES=4 python sd15_gen_eval.py
echo "=== score vs test/reals $(date) ==="
python eval_metrics.py --gen_dir "$GEN_OUT" --real_spec "FFBR:$D/branching_256/test/reals" --out "$GEN_OUT/metrics"
python - "$GEN_OUT/metrics/metrics.json" <<'PY'
import json, sys
o = json.load(open(sys.argv[1])).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"SD1.5 IN-DOMAIN (branching split): diversity={g('diversity'):.4f} realism={g('realism_vs_sibling'):.4f} CMMD={g('CMMD'):.3f}")
print("  compare -> Flux cn0.65+PG0.6: div 0.270 / real 0.524 / CMMD 8.78")
print("  old (WRONG) baseline generalist-SD1.5: div 0.266 / real 0.559 / CMMD 8.8-10.3")
PY
echo "=== DONE $(date) ==="
