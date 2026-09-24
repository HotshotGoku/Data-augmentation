#!/bin/bash
#SBATCH --job-name=strength_sweep
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Diversity-via-looser-conditioning gate: lower the ControlNet strength on SD1.5 mplex-ft
# (looser control -> source dominates less -> more diverse samples). Sweep strength, score
# diversity + realism vs the strength=1.0 baseline (realism_sib 0.412 / diversity 0.381).
# Lightweight (--gres=gpu:1) to slip under the account CPU cap. If a strength raises diversity
# without tanking realism, its synth goes to the downstream decode.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
MPLEX_FT=$REPO/rattray_runs/multiplexed_ft/lightning_logs/version_54549115/checkpoints/epoch=2-step=1448.ckpt
OUTROOT=$REPO/strength_sweep_out
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
cd "$REPO"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.tsv"
printf "strength\tCMMD\trealism_sib\trealism_src\tdiversity\tcopy_rate\n" > "$SUMMARY"
for S in 1.0 0.75 0.5 0.3; do
  tag="str${S}"; gen="$OUTROOT/$tag/gen"; rm -rf "$OUTROOT/$tag"; mkdir -p "$gen"
  echo "=== strength=$S $(date) ==="
  FT_EVAL_CKPT="$MPLEX_FT" INFER_STRENGTH="$S" python reptorep_infer_rattray.py --input_dir "$MPX/sources" --out_dir "$gen" || { echo "  gen FAILED $S"; continue; }
  python eval_metrics.py --gen_dir "$gen" --real_spec "MPLEX:$MPX/reals_heldout" --out "$OUTROOT/$tag" || { echo "  eval FAILED $S"; continue; }
  python - "$S" "$OUTROOT/$tag/metrics.json" >> "$SUMMARY" <<'PY'
import json, sys
s, mj = sys.argv[1], sys.argv[2]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"{s}\t{g('CMMD'):.3f}\t{g('realism_vs_sibling'):.4f}\t{g('realism_vs_source'):.4f}\t{g('diversity'):.4f}\t{g('copy_rate')}")
PY
done
echo "=== SD1.5 mplex-ft control-strength sweep (str1.0 = baseline realism_sib 0.412 / diversity 0.381) ==="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k5 -gr; } | column -t -s $'\t'
echo "=== DONE $(date) ==="
