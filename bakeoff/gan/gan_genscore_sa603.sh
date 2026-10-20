#!/bin/bash
#SBATCH --job-name=gan_genscore
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh

# --- generate (lwgan env) ---
conda activate /hpc/group/youlab/sa603/envs/lwgan
cd "$REPO"
OUT=$REPO/gan_runs/gen/lwgan_m${LOAD:-2}
LOAD=${LOAD:-2} N=${N:-400} OUT="$OUT" python gan_generate.py

# --- score (pytorch_PA env; separate from generator) ---
conda deactivate; conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
echo "=== score lwgan (unconditional) vs test/reals $(date) ==="
python eval_metrics.py --gen_dir "$OUT" --real_spec "FFBR:$D/branching_256/test/reals" --out "$OUT/metrics" 2>&1 | tail -3 || echo "eval_metrics note (realism/diversity may be n/a for unconditional; CMMD is the comparable metric)"
python - "$OUT/metrics/metrics.json" <<'PY'
import json, sys, os
try:
    o = json.load(open(sys.argv[1])).get("overall", {})
    g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
    print(f"\n>>> lightweight-GAN (FastGAN, ~25k steps, UNCONDITIONAL): CMMD={g('CMMD'):.2f} diversity={g('diversity')}")
except Exception as e:
    print("metrics parse:", e)
print(">>> REFERENCE: in-domain SD1.5 CMMD 3.38 | Flux-bf16 4.84 | PixCell-branching ~50 (washed)")
PY
echo "=== DONE $(date) ==="
