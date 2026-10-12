#!/bin/bash
#SBATCH --job-name=mux_cond_ddpm
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Fill matrix cell: from-scratch conditional DDPM on MULTIPLEXED. train -> gen -> score (one job).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
ENV=/hpc/group/youlab/sa603/envs/ddpm
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"; cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
OUT=$REPO/cond_ddpm_runs/mux

echo "=== train conditional DDPM mux $(date) STEPS=${STEPS:-40000} ==="
MODE=train DATA_JSON=$MPX/train_multiplexed.json STEPS=${STEPS:-40000} OUT="$OUT" python conditional_ddpm.py

echo "=== gen on mux sources (DDIM 50) $(date) ==="
MODE=gen CKPT="$OUT/ema_final.pth" OUT="$OUT/gen" SOURCES=$MPX/sources \
  NUM=4 STEPS_SAMPLE=50 ETA=0.0 python conditional_ddpm.py

echo "=== score (pytorch_PA) $(date) ==="
conda deactivate; conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python eval_metrics.py --gen_dir "$OUT/gen" --real_spec "MPLEX:$MPX/reals_heldout" --out "$OUT/gen/metrics" || echo "score note"
python - "$OUT/gen/metrics/metrics.json" <<'PY'
import json, sys
o=json.load(open(sys.argv[1])).get("overall",{}); g=lambda k:(o[k]["mean"] if isinstance(o.get(k),dict) else o.get(k))
print(f">>> COND DDPM (mux): realism={g('realism_vs_sibling'):.3f} CMMD={g('CMMD'):.2f} diversity={g('diversity'):.3f}")
print(">>> REF mux: SD1.5 in-domain CMMD 12.6 | PixCell 5.78")
PY
echo "=== DONE $(date) ==="
