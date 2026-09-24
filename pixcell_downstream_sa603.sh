#!/bin/bash
#SBATCH --job-name=pixcell_ds
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=7:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Downstream decode for a PixCell fine-tune epoch: generate synth (pixcell env), then run the
# aug_pixcell_epEP arm (K=1,2,3 x 5 seeds, resnet50/80ep to match results_v3), compare vs the
# existing v3 arms. Submit per epoch:  sbatch --export=ALL,EP=20  and  --export=ALL,EP=30
set -uo pipefail
: "${EP:?set EP (e.g. 20 or 30)}"
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
CKPT=$REPO/pixcell_cn_runs/mplex/controlnet_ep${EP}.pth
SYNTH=$MPX/synth_pixcell_ep${EP}
V3=$REPO/downstream_multiplexed_out/results_v3.tsv
RES=$REPO/downstream_multiplexed_out/results_pixcell_ep${EP}.tsv
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
[ -f "$CKPT" ] || { echo "FATAL: no ckpt $CKPT"; exit 1; }

echo "############## PHASE A: gen synth (pixcell env) EP=$EP $(date) ##############"
conda activate /hpc/group/youlab/sa603/envs/pixcell
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export PYTHONPATH=/hpc/group/youlab/sa603/code/PixCell/controlnet:${PYTHONPATH:-}
cd "$REPO"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
if [ "$(ls "$SYNTH"/*.png 2>/dev/null | wc -l)" -lt 1600 ]; then
  PIXCELL_CN_CKPT="$CKPT" SYNTH_DIR="$SYNTH" SYNTH_PER_SRC=8 STEPS=30 GUID=1.0 python pixcell_gen_synth.py || { echo "gen FAILED"; exit 1; }
fi
echo "synth count: $(ls "$SYNTH"/*.png 2>/dev/null | wc -l)"
conda deactivate

echo "############## PHASE B: decode arm aug_pixcell_ep${EP} (pytorch_PA env) $(date) ##############"
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
rm -f "$RES"
for K in 1 2 3; do
  for SEED in 0 1 2 3 4; do
    echo "--- K=$K seed=$SEED $(date) ---"
    env K=$K MODE=augmenter MODE_TAG=aug_pixcell_ep${EP} SEED=$SEED N_PER=8 EPOCHS=80 BACKBONE=resnet50 \
      SYNTH_DIR="$SYNTH" RESULTS_TSV="$RES" python downstream_multiplexed_decode.py || echo "  FAIL K=$K seed=$SEED"
  done
done

echo "############## COMPARISON (mean over seeds; higher=better) $(date) ##############"
python - "$V3" "$REPO"/downstream_multiplexed_out/results_pixcell_ep*.tsv <<'PY'
import csv, collections, statistics, sys, glob
files = [sys.argv[1]] + sorted(set(sys.argv[2:]))
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for fn in files:
    try: f = open(fn)
    except FileNotFoundError: continue
    for d in csv.DictReader(f, delimiter='\t'):
        key = (d['mode'], d['K'])
        for m in ('joint_acc', 'atc_r2', 'iptg_r2'):
            try: rows[key][m].append(float(d[m]))
            except Exception: pass
print(f"{'arm':16s} {'K':>3s} {'n':>2s} {'joint_acc':>10s} {'atc_r2':>8s} {'iptg_r2':>8s}")
for (mode, K) in sorted(rows, key=lambda x: (x[1], x[0])):
    r = rows[(mode, K)]; n = len(r['joint_acc'])
    g = lambda m: (statistics.mean(r[m]) if r[m] else float('nan'))
    print(f"{mode:16s} {K:>3s} {n:>2d} {g('joint_acc'):>10.3f} {g('atc_r2'):>8.3f} {g('iptg_r2'):>8.3f}")
PY
echo "=== DONE EP=$EP $(date) ==="
