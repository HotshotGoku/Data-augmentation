#!/bin/bash
#SBATCH --job-name=rat_knobsweep
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# #3 Inference-knob sweep (Supp Fig 16) on the BEST fine-tuned Rattray model.
# Sweeps control strength x guess_mode x guidance scale, scored on the held-out Rattray test.
# Inference-only: NO checkpoints written (storage-cheap).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
RAT=/hpc/group/youlab/sa603/data/external_datasets/rattray_2023
TESTSRC=$RAT/test_sources; TESTREAL=$RAT/test_reals
OUTROOT=$REPO/rattray_knob_sweep
BEST=$(find $REPO/rattray_runs/rat_none_1e-5 -name "*.ckpt" | head -1)

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
[ -n "$BEST" ] || { echo "FATAL: no rat_none_1e-5 ckpt"; exit 1; }
echo "=== knobsweep on $BEST ==="

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.tsv"
printf "strength\tguess\tscale\tFID\tdiversity\trealism_vs_sibling\tbaseline_real_vs_real\n" > "$SUMMARY"

for strength in 0.85 1.0 1.25 1.5; do
  for guess in 0 1; do
    for scale in 9.0 15.1; do
      tag="s${strength}_g${guess}_c${scale}"; gen="$OUTROOT/$tag/gen"; mkdir -p "$gen"
      echo "=== [$tag] ==="
      FT_EVAL_CKPT="$BEST" INFER_STRENGTH="$strength" INFER_GUESS="$guess" INFER_SCALE="$scale" \
        python reptorep_infer_rattray.py --input_dir "$TESTSRC" --out_dir "$gen" || { echo "  infer FAILED"; continue; }
      python eval_metrics.py --gen_dir "$gen" --real_spec "RATTRAY:$TESTREAL" --out "$OUTROOT/$tag" || { echo "  eval FAILED"; continue; }
      python - "$strength" "$guess" "$scale" "$OUTROOT/$tag/metrics.json" >> "$SUMMARY" <<'PY'
import json, sys
st, g, sc, mj = sys.argv[1:5]
o = json.load(open(mj)).get("overall", {})
val = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"{st}\t{g}\t{sc}\t{o.get('FID')}\t{val('diversity')}\t{val('realism_vs_sibling')}\t{val('baseline_real_vs_real')}")
PY
      rm -rf "$gen"   # storage-cheap: drop generated images after scoring
    done
  done
done
echo "=== DONE — summary ==="; column -t -s $'\t' "$SUMMARY" | sort -t$'\t' -k6 -g
