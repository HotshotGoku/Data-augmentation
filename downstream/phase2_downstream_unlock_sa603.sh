#!/bin/bash
#SBATCH --job-name=ds_unlock
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Phase 2 downstream: does the decoder-unfrozen augmenter (mplex_unlock ep4, realism_sib 0.381)
# improve the multiplexed aTc/IPTG readout? Generate synth_unlock, run ONLY the aug_unlock arm
# (K=1,2,3 x 5 seeds) with the SAME config as results_v3 (resnet50, 80ep, N_PER=8, default labels),
# then print a comparison vs the existing v3 arms (real / classical / aug_mplexft / aug_general).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
UNLOCK=$REPO/rattray_runs/mplex_unlock/lightning_logs/version_55755686/checkpoints/epoch=4-step=2414.ckpt
SYNTH_UNLOCK=$MPX/synth_unlock
V3=$REPO/downstream_multiplexed_out/results_v3.tsv
RES=$REPO/downstream_multiplexed_out/results_unlock.tsv

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
[ -f "$UNLOCK" ] || { echo "FATAL: no unlock ckpt $UNLOCK"; exit 1; }

echo "=== gen synth_unlock (mplex_unlock ep4) $(date) ==="
if [ "$(ls "$SYNTH_UNLOCK"/*.png 2>/dev/null | wc -l)" -lt 1600 ]; then
  SYNTH_CKPT="$UNLOCK" SYNTH_DIR="$SYNTH_UNLOCK" SYNTH_PER_SRC=8 python gen_multiplexed_synth.py || { echo "gen FAILED"; exit 1; }
fi
echo "synth_unlock count: $(ls "$SYNTH_UNLOCK"/*.png 2>/dev/null | wc -l)"

rm -f "$RES"
echo "=== aug_unlock arm (resnet50, 80ep, N_PER=8) $(date) ==="
for K in 1 2 3; do
  for SEED in 0 1 2 3 4; do
    echo "--- K=$K seed=$SEED $(date) ---"
    env K=$K MODE=augmenter MODE_TAG=aug_unlock SEED=$SEED N_PER=8 EPOCHS=80 BACKBONE=resnet50 \
      SYNTH_DIR="$SYNTH_UNLOCK" RESULTS_TSV="$RES" \
      python downstream_multiplexed_decode.py || echo "  FAIL K=$K seed=$SEED"
  done
done

echo "=== COMPARISON (mean over seeds; joint_acc & R2: higher=better) $(date) ==="
python - "$V3" "$RES" <<'PY'
import csv, collections, statistics, sys
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for fn in sys.argv[1:]:
    try: f = open(fn)
    except FileNotFoundError: continue
    for d in csv.DictReader(f, delimiter='\t'):
        key = (d['mode'], d['K'])
        for m in ('joint_acc', 'atc_r2', 'iptg_r2'):
            try: rows[key][m].append(float(d[m]))
            except Exception: pass
print(f"{'arm':13s} {'K':>3s} {'n':>2s} {'joint_acc':>10s} {'atc_r2':>8s} {'iptg_r2':>8s}")
for (mode, K) in sorted(rows, key=lambda x: (x[1], x[0])):
    r = rows[(mode, K)]; n = len(r['joint_acc'])
    g = lambda m: (statistics.mean(r[m]) if r[m] else float('nan'))
    print(f"{mode:13s} {K:>3s} {n:>2d} {g('joint_acc'):>10.3f} {g('atc_r2'):>8.3f} {g('iptg_r2'):>8.3f}")
PY
echo "=== DONE $(date) ==="
