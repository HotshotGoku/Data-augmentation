#!/bin/bash
#SBATCH --job-name=cads_sweep_rat
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Phase 1 CADS confirmation on the RATTRAY canonical best model (rat_epochcurve epoch=3).
# Second domain (compact P. aeruginosa colonies, near-domain) to confirm the multiplexed CADS
# finding generalizes. Same 5 arms as the multiplexed sweep. Fast (13 sources x 2).
# NOTE: Rattray test set is tiny (13 strains) so CMMD/FID are noisy; realism_sib is the key read.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
RAT=/hpc/group/youlab/sa603/data/external_datasets/rattray_2023
TESTSRC=$RAT/test_sources; TESTREAL=$RAT/test_reals
OUTROOT=$REPO/cads_sweep_rattray_out
RAT_CKPT=$REPO/rattray_runs/rat_epochcurve/lightning_logs/version_54107493/checkpoints/epoch=3-step=719.ckpt

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
[ -f "$RAT_CKPT" ] || { echo "FATAL: no rattray ckpt at $RAT_CKPT"; exit 1; }

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.tsv"
printf "arm\tCMMD\tFID\trealism_sib\trealism_src\tdiversity\tcopy_rate\n" > "$SUMMARY"

score () {  # $1=tag  $2=s  $3=tau1  $4=tau2  $5=eta
  local tag="$1" gen res
  gen="$OUTROOT/$tag/gen"; res="$OUTROOT/$tag"
  echo "=== [$tag]  s=$2 tau1=$3 tau2=$4 eta=$5  $(date) ==="
  rm -rf "$res"; mkdir -p "$gen"
  export CADS_S="$2" CADS_TAU1="$3" CADS_TAU2="$4" INFER_ETA="$5" CADS_RESCALE=1 CADS_UNCOND=1
  FT_EVAL_CKPT="$RAT_CKPT" python gen_cads.py --input_dir "$TESTSRC" --out_dir "$gen" || { echo "  gen FAILED"; return; }
  python eval_metrics.py --gen_dir "$gen" --real_spec "RATTRAY:$TESTREAL" --out "$res" || { echo "  eval FAILED"; return; }
  python - "$tag" "$res/metrics.json" >> "$SUMMARY" <<'PY'
import json, sys
tag, mj = sys.argv[1], sys.argv[2]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"{tag}\t{g('CMMD')}\t{g('FID')}\t{g('realism_vs_sibling')}\t{g('realism_vs_source')}\t{g('diversity')}\t{g('copy_rate')}")
PY
}

score baseline 0.10 0.6 0.0 0.0    # tau2<=tau1 -> CADS off (Rattray canonical baseline)
score eta05    0.10 0.6 0.0 0.5    # plain DDIM stochasticity
score cads_s05 0.05 0.6 0.9 0.0
score cads_s10 0.10 0.6 0.9 0.0
score cads_s20 0.20 0.6 0.9 0.0

echo "=== DONE $(date) — sorted by realism_sib (lower=better) ==="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k4 -g; } | column -t -s $'\t'
