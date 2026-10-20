#!/bin/bash
#SBATCH --job-name=mux_flux_score
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Score Flux mux generations vs mux reals_heldout (pytorch_PA; separate from gen).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
G=$REPO/flux_cn_runs/mux/eval_cn10
echo "=== score $G $(date) ==="
python eval_metrics.py --gen_dir "$G" --real_spec "MPLEX:$MPX/reals_heldout" --out "$G/metrics" || echo "score note"
python - "$G/metrics/metrics.json" <<'PY'
import json, sys
o=json.load(open(sys.argv[1])).get("overall",{}); g=lambda k:(o[k]["mean"] if isinstance(o.get(k),dict) else o.get(k))
print(f">>> FLUX fp8/CN4 (mux): realism={g('realism_vs_sibling'):.3f} CMMD={g('CMMD'):.2f} diversity={g('diversity'):.3f}")
print(">>> REF mux: SD1.5 in-domain CMMD 12.6 | PixCell 5.78")
PY
echo "=== DONE $(date) ==="
