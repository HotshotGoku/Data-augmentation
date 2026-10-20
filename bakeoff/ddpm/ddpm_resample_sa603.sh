#!/bin/bash
#SBATCH --job-name=ddpm_resample
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
# Rule out undersampling: re-gen the TRAINED conditional DDPM with 250 DDIM steps (vs 50) -> rescore.
# If realism stays ~0.5, the blur is a training/data-prior limit, not a sampling artifact.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
ENV=/hpc/group/youlab/sa603/envs/ddpm
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"; cd "$REPO"
OUT=$REPO/cond_ddpm_runs
echo "=== re-gen 250 DDIM steps $(date) ==="
MODE=gen CKPT="$OUT/ema_final.pth" OUT="$OUT/gen250" SOURCES=$D/branching_256/test/sources \
  NUM=4 STEPS_SAMPLE=250 ETA=0.0 python conditional_ddpm.py
echo "=== score $(date) ==="
conda deactivate; conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python eval_metrics.py --gen_dir "$OUT/gen250" --real_spec "FFBR:$D/branching_256/test/reals" --out "$OUT/gen250/metrics" >/dev/null 2>&1 || echo "score note"
python - "$OUT/gen250/metrics/metrics.json" <<'PY'
import json, sys
o=json.load(open(sys.argv[1])).get("overall",{}); g=lambda k:(o[k]["mean"] if isinstance(o.get(k),dict) else o.get(k))
print(f">>> DDPM 250-STEP: realism={g('realism_vs_sibling'):.3f} CMMD={g('CMMD'):.2f} diversity={g('diversity'):.3f}")
print(">>> was 50-step: 0.538/9.81/0.213 | SD1.5 0.324/3.38/0.252 | real-floor ~0.25")
PY
echo "=== DONE $(date) ==="
