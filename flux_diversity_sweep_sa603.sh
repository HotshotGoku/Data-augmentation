#!/bin/bash
#SBATCH --job-name=flux_divsweep
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Diversity free-lever sweep: controlnet_conditioning_scale + guidance_scale. Two phases (gen then score).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
CKPT=$REPO/flux_cn_runs/branching_v2/controlnet_final.pth
ROOT=$REPO/flux_cn_runs/divsweep
export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
# tag:CN_SCALE:GUID
CONFIGS="base:1.0:3.5 cn085:0.85:3.5 cn07:0.7:3.5 cn05:0.5:3.5 g2:1.0:2.0 g1:1.0:1.0"

echo "############## PHASE A: generate (flowmatch) $(date) ##############"
conda activate /hpc/group/youlab/sa603/envs/flowmatch; cd "$REPO"
for c in $CONFIGS; do
  tag=${c%%:*}; rest=${c#*:}; scale=${rest%%:*}; guid=${rest#*:}
  echo "=== gen $tag (cn_scale=$scale guid=$guid) $(date +%H:%M:%S) ==="
  CN_CKPT="$CKPT" GEN_OUT="$ROOT/$tag" CN_LAYERS=2 RES=256 NUM_SAMPLES=6 MAX_SRC=8 \
    CN_SCALE=$scale GUID=$guid STEPS=28 OFFLOAD=none python flux_gen_eval.py || echo "  gen FAILED $tag"
done
conda deactivate

echo "############## PHASE B: score (pytorch_PA) $(date) ##############"
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
for c in $CONFIGS; do
  tag=${c%%:*}
  python eval_metrics.py --gen_dir "$ROOT/$tag" --real_spec "FFBR:/hpc/group/youlab/sa603/data/branching_256/test/reals" --out "$ROOT/$tag/metrics" >/dev/null 2>&1 && \
  python - "$tag" "$ROOT/$tag/metrics/metrics.json" <<'PY'
import json, sys
o = json.load(open(sys.argv[2])).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"DIVRESULT {sys.argv[1]}: diversity={g('diversity'):.4f} realism={g('realism_vs_sibling'):.4f} CMMD={g('CMMD'):.3f}")
PY
done
echo "=== baseline (full v2): diversity 0.220 realism 0.523 CMMD 6.52 | SD1.5: realism 0.559 ==="
echo "=== DONE $(date) ==="
