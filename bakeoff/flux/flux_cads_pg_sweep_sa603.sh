#!/bin/bash
#SBATCH --job-name=flux_cadspg
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# CADS + Particle Guidance diversity sweep (flux_gen_diverse.py), then score each.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
CKPT=$REPO/flux_cn_runs/branching_v2/controlnet_final.pth
ROOT=$REPO/flux_cn_runs/cadspg
export HF_HOME=/hpc/group/youlab/sa603/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
# tag:METHOD:CADS_NOISE:PG_ALPHA
CONFIGS="nonebatch:none:0:0 cads05:cads:0.5:0 cads10:cads:1.0:0 cads20:cads:2.0:0 pg1:pg:0:1.0 pg4:pg:0:4.0 both:both:1.0:1.0"

echo "############## PHASE A: generate (flowmatch) $(date) ##############"
conda activate /hpc/group/youlab/sa603/envs/flowmatch; cd "$REPO"
for c in $CONFIGS; do
  IFS=: read -r tag method noise alpha <<< "$c"
  echo "=== gen $tag (method=$method noise=$noise alpha=$alpha) $(date +%H:%M:%S) ==="
  CN_CKPT="$CKPT" GEN_OUT="$ROOT/$tag" METHOD=$method CADS_NOISE=$noise PG_ALPHA=$alpha \
    CN_LAYERS=2 RES=256 NUM_SAMPLES=4 MAX_SRC=6 STEPS=28 GUID=3.5 CN_SCALE=1.0 OFFLOAD=none \
    python flux_gen_diverse.py || echo "  gen FAILED $tag"
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
echo "=== baseline (full v2): diversity 0.220 realism 0.523 CMMD 6.52 ==="
echo "=== DONE $(date) ==="
