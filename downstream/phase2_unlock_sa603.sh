#!/bin/bash
#SBATCH --job-name=mplex_unlock
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Phase 2: unfreeze the SD decoder (sd_locked=False) and fine-tune on multiplexed pairs FROM BASE.
# Same recipe as the original mplex-ft (lr 1e-5, 6 epochs, dihedral) EXCEPT sd_locked=False and
# batch 2 x accum 2 (= effective batch 4; sd_locked=False OOMs at batch 4). Per-epoch ckpts, then
# score every epoch vs the held-out real pool (identical contract to multiplexed_ft_eval) to get
# the epoch curve. Baseline to beat: mplex_ft_epoch=2 = realism_sib 0.4117 / CMMD 12.57.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
BASE=$REPO/lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt
TAG=mplex_unlock
OUTROOT=$REPO/phase2_unlock_out

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export WANDB_MODE=offline
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

echo "=== TRAIN (sd_locked=False, base->multiplexed) $(date) ==="
FT_JSON="$MPX/train_multiplexed.json" FT_RESUME="$BASE" FT_TAG="$TAG" \
  FT_LR=1e-5 FT_EPOCHS=6 FT_SEED=42 FT_FREEZE=none FT_SAVE_TOPK=-1 \
  FT_SDLOCKED=0 FT_BATCH=2 FT_ACCUM=2 RATTRAY_AUG=1 \
  python reptorep_finetune_rattray.py || { echo "TRAIN FAILED"; exit 1; }

echo "=== EVAL every epoch vs held-out reals $(date) ==="
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.tsv"
printf "model\tCMMD\tFID\trealism_sib\trealism_src\tdiversity\tcopy_rate\n" > "$SUMMARY"

score () {  # $1=tag  $2=ckpt
  local tag="$1" ckpt="$2" gen res
  gen="$OUTROOT/$tag/gen"; res="$OUTROOT/$tag"
  echo "=== [$tag] ==="; rm -rf "$res"; mkdir -p "$gen"
  FT_EVAL_CKPT="$ckpt" python reptorep_infer_rattray.py --input_dir "$MPX/sources" --out_dir "$gen" || { echo "  infer FAILED"; return; }
  python eval_metrics.py --gen_dir "$gen" --real_spec "MPLEX:$MPX/reals_heldout" --out "$res" || { echo "  eval FAILED"; return; }
  python - "$tag" "$res/metrics.json" >> "$SUMMARY" <<'PY'
import json, sys
tag, mj = sys.argv[1], sys.argv[2]
o = json.load(open(mj)).get("overall", {})
g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
print(f"{tag}\t{g('CMMD')}\t{g('FID')}\t{g('realism_vs_sibling')}\t{g('realism_vs_source')}\t{g('diversity')}\t{g('copy_rate')}")
PY
  rm -rf "$gen"   # keep ckpts for manual pruning; drop gen images (storage)
}

while IFS= read -r c; do
  ep=$(echo "$c" | grep -oE 'epoch=[0-9]+')
  score "unlock_$ep" "$c"
done < <(find "$REPO/rattray_runs/$TAG" -name "*.ckpt" 2>/dev/null | sort)

echo "=== DONE $(date) — sorted by realism_sib (lower=better; baseline mplex_ft_epoch=2 = 0.4117 / 12.57) ==="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k4 -g; } | column -t -s $'\t'
