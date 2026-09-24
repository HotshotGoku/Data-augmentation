#!/bin/bash
#SBATCH --job-name=pc1024_gen
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Generate PixCell-1024 replicates for scoring. Defaults to the final fine-tuned ControlNet.
# Usage: sbatch --export=ALL,SAMPLER=ddpm,NUM_SAMPLES=8 pixcell1024_gen_eval_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export PYTHONPATH=/hpc/group/youlab/sa603/code/PixCell/controlnet:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1  # live progress logs
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/pixcell
cd "$REPO"
TAG=${TAG:-mplex}
case "$TAG" in
  branching) DEF_SOURCES=/hpc/group/youlab/sa603/data/branching_1024/test/sources ;;
  *)         DEF_SOURCES=/hpc/group/youlab/sa603/data/multiplexed_eval/sources ;;
esac
SOURCES=${SOURCES:-$DEF_SOURCES}
CKPT=${PIXCELL_CN_CKPT:-$REPO/pixcell1024_cn_runs/$TAG/controlnet_final.pth}
OUT=${GEN_OUT:-$REPO/pixcell1024_gen/${TAG}_${SAMPLER:-ddim}}
echo "=== node/GPU $(date) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
echo "=== gen start TAG=$TAG CKPT=$CKPT SOURCES=$SOURCES OUT=$OUT SAMPLER=${SAMPLER:-ddim} N=${NUM_SAMPLES:-2} ==="
PIXCELL_CN_CKPT="$CKPT" GEN_OUT="$OUT" SOURCES="$SOURCES" python pixcell1024_gen_eval.py
echo "=== gen exit=$? $(date) count=$(ls "$OUT"/*.png 2>/dev/null | wc -l) ==="

echo "=== PHASE B: score (pytorch_PA env) $(date) ==="
case "$TAG" in
  branching) REAL_SPEC="FFBR:/hpc/group/youlab/sa603/data/branching_256/test/reals" ;;
  *)         REAL_SPEC="MPLEX:/hpc/group/youlab/sa603/data/multiplexed_eval/reals_heldout" ;;
esac
conda deactivate
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
python eval_metrics.py --gen_dir "$OUT" --real_spec "$REAL_SPEC" --out "$OUT/metrics" && \
python - "${TAG}-1024" "$OUT/metrics/metrics.json" <<'PY'
import json, sys
tag, mj = sys.argv[1], sys.argv[2]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"RESULT {tag}: CMMD={g('CMMD')} realism_sib={g('realism_vs_sibling')} diversity={g('diversity')} copy_rate={g('copy_rate')} realism_src={g('realism_vs_source')}")
PY
echo "=== DONE $(date) ==="
