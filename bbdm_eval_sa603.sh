#!/bin/bash
#SBATCH --job-name=bbdm_eval
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Score the trained LBBDM on the frozen multiplexed contract.
# Phase A (bbdm env): --sample_to_eval -> per-source output_{j}.png; remap output_0/1 -> {prefix}_{1,2}.png.
# Phase B (pytorch_PA env): eval_metrics v2 vs reals_heldout.
set -uo pipefail
BBDM=/hpc/group/youlab/sa603/code/BBDM
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
OUTROOT=$REPO/bbdm_eval_out
CKPT=${BBDM_CKPT:-$BBDM/results/mplex/LBBDM-f4/checkpoint/top_model_epoch_98.pth}
TAG=${TAG:-bbdm_top98}
GEN=$OUTROOT/$TAG/gen
mkdir -p "$GEN"
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh

echo "############## PHASE A: BBDM sample_to_eval (bbdm env) $(date) ##############"
conda activate /hpc/group/youlab/sa603/envs/bbdm
cd "$BBDM"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
python main.py --config configs/youlab-LBBDM-f4.yaml --sample_to_eval --resume_model "$CKPT" --gpu_ids 0
echo "=== remap output_{0,1}.png -> {prefix}_{1,2}.png ==="
find "results/${DS_NAME:-mplex}/LBBDM-f4" -name "output_0.png" | while read -r f; do
  d=$(dirname "$f"); sid=$(basename "$d"); sid=${sid%.png}
  cp "$d/output_0.png" "$GEN/${sid}_1.png" 2>/dev/null || true
  [ -f "$d/output_1.png" ] && cp "$d/output_1.png" "$GEN/${sid}_2.png" 2>/dev/null || true
done
echo "gen count: $(ls "$GEN"/*.png 2>/dev/null | wc -l)  (expect ~140 = 70 sources x 2)"
conda deactivate

echo "############## PHASE B: score (pytorch_PA env) $(date) ##############"
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
cd "$REPO"
python eval_metrics.py --gen_dir "$GEN" --real_spec "MPLEX:$MPX/reals_heldout" --out "$OUTROOT/$TAG" || { echo 'eval FAILED'; exit 1; }
python - "$TAG" "$OUTROOT/$TAG/metrics.json" <<'PY'
import json, sys
tag, mj = sys.argv[1], sys.argv[2]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"\n=== BBDM RESULT (baseline SD1.5 mplex-ft = realism_sib 0.4117 / CMMD 12.57 / div 0.381) ===")
print(f"{tag}: CMMD={g('CMMD'):.4f} FID={g('FID'):.2f} realism_sib={g('realism_vs_sibling'):.4f} "
      f"realism_src={g('realism_vs_source'):.4f} diversity={g('diversity'):.4f} copy_rate={g('copy_rate')}")
PY
echo "=== DONE $(date) ==="
