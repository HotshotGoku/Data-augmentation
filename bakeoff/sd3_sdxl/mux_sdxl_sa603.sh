#!/bin/bash
#SBATCH --job-name=mux_sdxl
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Fill matrix cell: zero-shot SDXL img2img on MULTIPLEXED sources. Gen (flowmatch) -> score (pytorch_PA).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
echo "=== zero-shot SDXL on mux gen $(date) ==="
NAME=sdxl_mux MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0 DTYPE=fp16 STRENGTHS=0.7 NUM=4 \
  SOURCES=$MPX/sources OUTROOT=$REPO/backbone_screen RES=256 python screen_img2img.py

echo "=== score vs mux reals_heldout (pytorch_PA) $(date) ==="
conda deactivate; conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
G=$REPO/backbone_screen/sdxl_mux_s70
python eval_metrics.py --gen_dir "$G" --real_spec "MPLEX:$MPX/reals_heldout" --out "$G/metrics" || echo "score note"
python - "$G/metrics/metrics.json" <<'PY'
import json,sys
o=json.load(open(sys.argv[1])).get("overall",{}); g=lambda k:(o[k]["mean"] if isinstance(o.get(k),dict) else o.get(k))
print(f">>> ZERO-SHOT SDXL (mux): realism={g('realism_vs_sibling'):.3f} CMMD={g('CMMD'):.2f} diversity={g('diversity'):.3f}")
print(">>> REF mux: SD1.5 in-domain CMMD 12.6 | PixCell 5.78")
PY
echo "=== DONE $(date) ==="
