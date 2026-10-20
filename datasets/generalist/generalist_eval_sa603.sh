#!/bin/bash
#SBATCH --job-name=gen_eval
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Per-domain realism: BASE vs every GENERALIST epoch, on held-out conditions from each domain.
# Question: does ONE jointly-fine-tuned model become faithful across ALL of Kinshuk's colony domains?
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
GEN=/hpc/group/youlab/sa603/data/generalist
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
OUTROOT=$REPO/generalist_eval_out
BASE=$REPO/lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.tsv"
printf "model\tdomain\tFID\tdiversity\trealism_vs_sibling\tbaseline_real_vs_real\n" > "$SUMMARY"

declare -A SRC REAL
for code in pakp kl2 ff nlev selx; do SRC[$code]="$GEN/test/$code/sources"; REAL[$code]="$GEN/test/$code/reals"; done
SRC[mplex]="$MPX/sources"; REAL[mplex]="$MPX/reals_heldout"

score () {  # $1=tag $2=ckpt $3=domain
  local tag="$1" ckpt="$2" code="$3" gen="$OUTROOT/$1/$3/gen" res="$OUTROOT/$1/$3"
  rm -rf "$res"; mkdir -p "$gen"
  FT_EVAL_CKPT="$ckpt" python reptorep_infer_rattray.py --input_dir "${SRC[$code]}" --out_dir "$gen" || { echo "  infer FAIL $tag/$code"; return; }
  python eval_metrics.py --gen_dir "$gen" --real_spec "MPLEX:${REAL[$code]}" --out "$res" || { echo "  eval FAIL $tag/$code"; return; }
  python - "$tag" "$code" "$res/metrics.json" >> "$SUMMARY" <<'PY'
import json, sys
tag, code, mj = sys.argv[1], sys.argv[2], sys.argv[3]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"{tag}\t{code}\t{o.get('FID')}\t{g('diversity')}\t{g('realism_vs_sibling')}\t{g('baseline_real_vs_real')}")
PY
}

DOMAINS="pakp kl2 ff nlev selx mplex"
for code in $DOMAINS; do score "base" "$BASE" "$code"; done
while IFS= read -r c; do
  ep=$(echo "$c" | grep -oE 'epoch=[0-9]+')
  for code in $DOMAINS; do score "gen_$ep" "$c" "$code"; done
done < <(find "$REPO/rattray_runs/generalist" -name "*.ckpt" 2>/dev/null | sort)

echo "=== DONE $(date) — per-domain realism (lower=better; baseline col = real-vs-real floor) ==="
column -t -s $'\t' "$SUMMARY"
