#!/bin/bash
#SBATCH --job-name=rescore_v2
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=6:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Re-score all model families with metrics v2 (CMMD + nearest-neighbor per-sample), EVAL-ONLY:
# reuse existing gen/ (deterministic, already on disk) -> no regeneration.
# SMOKE=1 -> validate on multiplexed base only + back-compat check, then exit.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
GEN=/hpc/group/youlab/sa603/data/generalist
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
MPX_REAL="MPLEX:$MPX/reals_heldout"

eval_one () {  # $1=res_dir (contains gen/) $2=real_spec  -> writes metrics.json in res_dir
  local res="$1"
  [ "$(ls "$res"/gen/*.png 2>/dev/null | wc -l)" -ge 2 ] || { echo "  SKIP $(basename "$res") (no gen)"; return 1; }
  python eval_metrics.py --gen_dir "$res/gen" --real_spec "$2" --out "$res" >/dev/null 2>&1 \
    && echo "  ok $res" || { echo "  FAIL $res"; return 1; }
}
extract () {  # $1=res_dir $2=model $3=domain  -> one TSV line to stdout
  [ -f "$1/metrics.json" ] || return
  python - "$2" "$3" "$1/metrics.json" <<'PY'
import json, sys
m, d, mj = sys.argv[1], sys.argv[2], sys.argv[3]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) and o.get(k) else "")
print("\t".join(str(x) for x in [m, d, o.get("CMMD"), o.get("FID"), g("realism_vs_sibling"),
      g("realism_nn_lpips_vgg"), g("realism_nn_ssim"), g("realism_nn_orb"),
      o.get("copy_rate"), g("baseline_real_vs_real"), g("baseline_nn_real_vs_real")]))
PY
}

if [ "${SMOKE:-0}" = "1" ]; then
  echo "=== SMOKE: multiplexed base -> /tmp/smoke_base ==="
  python eval_metrics.py --gen_dir "$REPO/multiplexed_ft_eval_out/base/gen" --real_spec "$MPX_REAL" --out /tmp/smoke_base 2>&1 | tail -70
  echo "=== back-compat: committed base/metrics.json vs new ==="
  python - <<'PY'
import json
old = json.load(open("/hpc/group/youlab/sa603/code/Data_augmentation/multiplexed_ft_eval_out/base/metrics.json"))["overall"]
new = json.load(open("/tmp/smoke_base/metrics.json"))["overall"]
for k in ["FID", "diversity", "realism_vs_sibling", "realism_vs_source", "baseline_real_vs_real"]:
    print(f"{k:24s} OLD={old.get(k)}  NEW={new.get(k)}  {'MATCH' if old.get(k)==new.get(k) else 'DIFF'}")
print("new keys added:", [k for k in new if k not in old])
PY
  exit 0
fi

SUM="$REPO/metrics_v2_summary.tsv"
printf "family\tmodel\tdomain\tCMMD\tFID\trealism_sib\trealism_nn_vgg\tnn_ssim\tnn_orb\tcopy_rate\tbase_sib\tbase_nn\n" > "$SUM"

echo "=== multiplexed family ==="
for d in "$REPO"/multiplexed_ft_eval_out/*/; do
  m=$(basename "$d"); [ -d "${d}gen" ] || continue
  eval_one "${d%/}" "$MPX_REAL" && { printf "multiplexed\t"; extract "${d%/}" "$m" "mplex"; } >> "$SUM"
done
echo "=== generalist family ==="
for d in "$REPO"/generalist_eval_out/*/; do
  m=$(basename "$d")
  for dom in pakp kl2 ff nlev selx mplex; do
    sub="${d}${dom}"; [ -d "$sub/gen" ] || continue
    [ "$dom" = "mplex" ] && spec="$MPX_REAL" || spec="MPLEX:$GEN/test/$dom/reals"
    eval_one "$sub" "$spec" && { printf "generalist\t"; extract "$sub" "$m" "$dom"; } >> "$SUM"
  done
done
echo "=== warm-start family ==="
for d in "$REPO"/mplex_warmstart_eval_out/*/; do
  m=$(basename "$d"); [ -d "${d}gen" ] || continue
  eval_one "${d%/}" "$MPX_REAL" && { printf "warmstart\t"; extract "${d%/}" "$m" "mplex"; } >> "$SUM"
done

echo "=== DONE $(date) — metrics_v2_summary.tsv ==="; column -t -s $'\t' "$SUM"
