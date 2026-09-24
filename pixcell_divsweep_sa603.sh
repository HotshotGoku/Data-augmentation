#!/bin/bash
#SBATCH --job-name=pixcell_divsweep
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Cheap diversity gate: can a stochastic sampler push PixCell (ep20) diversity toward the SD1.5
# baseline (0.381) without wrecking realism? Gen the 70-source metrics set under 4 samplers, score.
# Lightweight (--gres=gpu:1) so it slips in under the account CPU cap. If a setting recovers
# diversity with realism still < 0.412, run the downstream decode with it.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
CKPT=$REPO/pixcell_cn_runs/mplex/controlnet_ep20.pth
OUTROOT=$REPO/pixcell_divsweep_out
mkdir -p "$OUTROOT"
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh

echo "############## PHASE A: gen under each sampler (pixcell env) $(date) ##############"
conda activate /hpc/group/youlab/sa603/envs/pixcell
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export PYTHONPATH=/hpc/group/youlab/sa603/code/PixCell/controlnet:${PYTHONPATH:-}
cd "$REPO"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
gen () {  # $1=tag $2=sampler $3=steps $4=eta
  echo "=== gen $1 (sampler=$2 steps=$3 eta=$4) $(date) ==="
  PIXCELL_CN_CKPT="$CKPT" GEN_OUT="$OUTROOT/$1/gen" NUM_SAMPLES=2 SAMPLER="$2" STEPS="$3" ETA="$4" GUID=1.0 \
    python pixcell_gen_eval.py || echo "  gen FAILED $1"
}
gen dpm_ctrl  dpm  50 0.0
gen sde50     sde  50 0.0
gen ddim50e1  ddim 50 1.0
gen ddpm100   ddpm 100 0.0
conda deactivate

echo "############## PHASE B: score (pytorch_PA env) $(date) ##############"
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
SUMMARY="$OUTROOT/summary.tsv"
printf "setting\tCMMD\trealism_sib\tdiversity\tcopy_rate\n" > "$SUMMARY"
for tag in dpm_ctrl sde50 ddim50e1 ddpm100; do
  [ -d "$OUTROOT/$tag/gen" ] || continue
  python eval_metrics.py --gen_dir "$OUTROOT/$tag/gen" --real_spec "MPLEX:$MPX/reals_heldout" --out "$OUTROOT/$tag" || continue
  python - "$tag" "$OUTROOT/$tag/metrics.json" >> "$SUMMARY" <<'PY'
import json, sys
tag, mj = sys.argv[1], sys.argv[2]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"{tag}\t{g('CMMD'):.3f}\t{g('realism_vs_sibling'):.4f}\t{g('diversity'):.4f}\t{g('copy_rate')}")
PY
done
echo "=== PixCell ep20 sampler diversity sweep (SD1.5 baseline div 0.381; ep20 dpm div 0.287) ==="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k4 -gr; } | column -t -s $'\t'
echo "=== DONE $(date) ==="
