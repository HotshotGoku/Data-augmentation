#!/bin/bash
#SBATCH --job-name=adapt_curve
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# WS2 data-efficiency: fine-tune on N training pairs, from BOTH base and generalist init, eval
# generation quality (CMMD + realism + per-sample) on the held-out test. Default dataset = Rattray
# (never seen by base OR generalist -> a clean few-shot test). Override paths via env for other datasets.
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
GENERALIST=$REPO/rattray_runs/generalist/lightning_logs/version_54592625/checkpoints/epoch=3-step=5723.ckpt

DS_TAG=${DS_TAG:-rattray}
FULL=${FT_JSON_FULL:-/hpc/group/youlab/sa603/data/external_datasets/rattray_2023/train_rattray.json}
SOURCES=${SOURCES:-/hpc/group/youlab/sa603/data/external_datasets/rattray_2023/test_sources}
REAL_SPEC=${REAL_SPEC:-RAT:/hpc/group/youlab/sa603/data/external_datasets/rattray_2023/test_reals}
NLIST=${NLIST:-"10 25 50 100 250 0"}   # 0 = all pairs
SEEDS=${SEEDS:-"0 1 2"}                 # error bars over data-subset + init seed
FT_LR=${FT_LR:-1e-5}; FT_EPOCHS=${FT_EPOCHS:-8}

OUT=$REPO/adapt_curve_out/$DS_TAG; mkdir -p "$OUT"
SUM=$REPO/adapt_curve_out/curve_${DS_TAG}.tsv
printf "init\tN\tseed\tn_pairs\tCMMD\trealism_sib\tbase_sib\tnn_vgg\n" > "$SUM"

run () {  # $1=init  $2=resume_ckpt  $3=N
  local init=$1 resume=$2 N=$3 seed=$4
  local tag=adapt_${DS_TAG}_${init}_N${N}_s${seed}
  local subj=$OUT/sub_${init}_N${N}_s${seed}.json
  FT_JSON_FULL=$FULL FT_JSON_SUBSET=$subj SUBSET_N=$N SUBSET_SEED=$seed python build_ft_subset.py || { echo "  subset FAIL $tag"; return; }
  local npairs=$(wc -l < "$subj")
  echo "=== $tag ($npairs pairs) $(date) ==="
  FT_JSON=$subj FT_RESUME=$resume FT_TAG=$tag FT_SEED=$seed FT_LR=$FT_LR FT_EPOCHS=$FT_EPOCHS FT_SAVE_TOPK=1 RATTRAY_AUG=1 \
    python reptorep_finetune_rattray.py || { echo "  FT FAIL $tag"; return; }
  local ckpt=$(find "$REPO/rattray_runs/$tag" -name "*.ckpt" 2>/dev/null | sort | tail -1)
  [ -z "$ckpt" ] && { echo "  no ckpt $tag"; return; }
  local gen=$OUT/$tag/gen; mkdir -p "$gen"
  FT_EVAL_CKPT="$ckpt" python reptorep_infer_rattray.py --input_dir "$SOURCES" --out_dir "$gen" \
    || { echo "  infer FAIL $tag"; rm -rf "$REPO/rattray_runs/$tag"; return; }
  python eval_metrics.py --gen_dir "$gen" --real_spec "$REAL_SPEC" --out "$OUT/$tag" \
    || { echo "  eval FAIL $tag"; rm -rf "$REPO/rattray_runs/$tag"; return; }
  python - "$init" "$N" "$seed" "$npairs" "$OUT/$tag/metrics.json" >> "$SUM" <<'PY'
import json, sys
init, N, seed, npairs, mj = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
o = json.load(open(mj))["overall"]
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) and o.get(k) else "")
print(f"{init}\t{N}\t{seed}\t{npairs}\t{o.get('CMMD')}\t{g('realism_vs_sibling')}\t{g('baseline_real_vs_real')}\t{g('realism_nn_lpips_vgg')}")
PY
  rm -rf "$REPO/rattray_runs/$tag"   # disk-safe: prune ckpt after scoring
}

for N in $NLIST; do
  for seed in $SEEDS; do
    run base "$BASE" "$N" "$seed"
    run generalist "$GENERALIST" "$N" "$seed"
  done
done
echo "=== DONE $(date) — curve_${DS_TAG}.tsv ==="; column -t -s $'\t' "$SUM"
