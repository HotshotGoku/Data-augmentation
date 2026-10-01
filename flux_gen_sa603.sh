#!/bin/bash
#SBATCH --job-name=flux_gen
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Flux pilot: generate replicates from a trained Flux ControlNet (flowmatch env), then score on the
# frozen bench (pytorch_PA env) vs SD1.5. Env: CN_CKPT, GEN_OUT, CN_LAYERS, RES, NUM_SAMPLES, MAX_SRC.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh

echo "=== PHASE A: generate (flowmatch) $(date) CKPT=$CN_CKPT OUT=$GEN_OUT ==="
conda activate /hpc/group/youlab/sa603/envs/flowmatch
cd "$REPO"
python flux_gen_eval.py
echo "=== gen done: $(ls "$GEN_OUT"/*.png 2>/dev/null | wc -l) imgs ==="

echo "=== PHASE B: score vs branching reals (pytorch_PA) $(date) ==="
conda deactivate; conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
python eval_metrics.py --gen_dir "$GEN_OUT" --real_spec "FFBR:/hpc/group/youlab/sa603/data/branching_256/test/reals" --out "$GEN_OUT/metrics" && \
python - "flux-branching" "$GEN_OUT/metrics/metrics.json" <<'PY'
import json, sys
o = json.load(open(sys.argv[2])).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"RESULT {sys.argv[1]}: CMMD={g('CMMD')} realism_sib={g('realism_vs_sibling')} diversity={g('diversity')} copy_rate={g('copy_rate')}")
print("SD1.5 branching reference: realism 0.559 / CMMD 8.8")
PY
echo "=== DONE $(date) ==="
