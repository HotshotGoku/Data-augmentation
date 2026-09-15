#!/bin/bash
#SBATCH --job-name=adapt_conv
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# WS-F convergence panel: directly rebut "small N trained fewer steps". For a SMALL N (25) and the
# FULL set (720), fine-tune from base saving EVERY epoch, then evaluate each epoch on the 13 held-out
# Rattray strains. If both plateau by ~epoch 3, adaptation is epoch-driven (not step-driven) and small
# N is NOT undertrained -> epochs is the fair unit for the data-efficiency curve.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

BASE=$REPO/lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt
FULL=/hpc/group/youlab/sa603/data/external_datasets/rattray_2023/train_rattray.json
SOURCES=/hpc/group/youlab/sa603/data/external_datasets/rattray_2023/test_sources
REAL_SPEC=RAT:/hpc/group/youlab/sa603/data/external_datasets/rattray_2023/test_reals
EPOCHS=${EPOCHS:-6}
NLIST=${NLIST:-"25 0"}   # 0 = all pairs (720)

OUT=$REPO/adapt_curve_out/convergence; mkdir -p "$OUT"
SUM=$REPO/adapt_curve_out/convergence_rattray.tsv
printf "N\tepoch\tn_pairs\tCMMD\trealism_sib\tbase_sib\n" > "$SUM"

for N in $NLIST; do
  seed=0
  tag=conv_N${N}_s${seed}
  subj=$OUT/sub_N${N}.json
  FT_JSON_FULL=$FULL FT_JSON_SUBSET=$subj SUBSET_N=$N SUBSET_SEED=$seed python build_ft_subset.py || { echo "subset FAIL N=$N"; continue; }
  npairs=$(wc -l < "$subj")
  echo "=== fine-tune N=$N ($npairs pairs), save every epoch $(date) ==="
  FT_JSON=$subj FT_RESUME=$BASE FT_TAG=$tag FT_SEED=$seed FT_LR=1e-5 FT_EPOCHS=$EPOCHS FT_SAVE_TOPK=-1 RATTRAY_AUG=1 \
    python reptorep_finetune_rattray.py || { echo "FT FAIL N=$N"; continue; }
  rundir=$REPO/rattray_runs/$tag
  for ckpt in $(find "$rundir" -name "*.ckpt" 2>/dev/null | sort -V); do
    ep=$(basename "$ckpt" | grep -oE "epoch=[0-9]+" | grep -oE "[0-9]+")
    gen=$OUT/${tag}_ep${ep}/gen; mkdir -p "$gen"
    FT_EVAL_CKPT="$ckpt" python reptorep_infer_rattray.py --input_dir "$SOURCES" --out_dir "$gen" || { echo "  infer FAIL N=$N ep=$ep"; continue; }
    python eval_metrics.py --gen_dir "$gen" --real_spec "$REAL_SPEC" --out "$OUT/${tag}_ep${ep}" || { echo "  eval FAIL N=$N ep=$ep"; continue; }
    python - "$N" "$ep" "$npairs" "$OUT/${tag}_ep${ep}/metrics.json" >> "$SUM" <<'PY'
import json, sys
N, ep, npairs, mj = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
o = json.load(open(mj))["overall"]
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) and o.get(k) else "")
print(f"{N}\t{ep}\t{npairs}\t{o.get('CMMD')}\t{g('realism_vs_sibling')}\t{g('baseline_real_vs_real')}")
PY
  done
  rm -rf "$rundir"   # disk-safe: prune all epoch ckpts after scoring
done
echo "=== DONE $(date) — convergence_rattray.tsv ==="; column -t -s $'\t' "$SUM"
