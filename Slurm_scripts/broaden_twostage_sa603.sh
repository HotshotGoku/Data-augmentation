#!/bin/bash
#SBATCH --job-name=broaden_2stage
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Two-stage broadening (uses ALL SwarmEvo data):
#   Stage 1: continue-train the base on the FULL SwarmEvo pair set  -> broadened base
#   Stage 2: fine-tune the broadened base on Rattray                -> final model
# Then compare vs Rattray-only (rat_epochcurve epoch3 = 1.69x) via sweep_rattray.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
BASEDATA=/hpc/group/youlab/sa603/data/external_datasets
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

echo "=== STAGE 1: broaden base on ALL SwarmEvo ($(date)) ==="
FT_JSON=$BASEDATA/swarmevo/train_swarmevo_all.json FT_TAG=broaden_full \
  FT_FREEZE=none FT_LR=1e-5 FT_EPOCHS=${BROADEN_EPOCHS:-2} FT_SAVE_TOPK=1 \
  python reptorep_finetune_rattray.py   # FT_RESUME unset -> defaults to base CKPT_PATH_V4
s1=$?; [ $s1 -eq 0 ] || { echo "stage1 failed ($s1)"; exit $s1; }

BROADENED=$(find rattray_runs/broaden_full -name "*.ckpt" | sort | tail -1)
echo "=== broadened ckpt: $BROADENED ==="
[ -n "$BROADENED" ] || { echo "no broadened ckpt"; exit 1; }

echo "=== STAGE 2: fine-tune broadened base on Rattray ($(date)) ==="
FT_JSON=$BASEDATA/rattray_2023/train_rattray.json FT_RESUME="$BROADENED" FT_TAG=rattray_after_broaden \
  FT_FREEZE=none FT_LR=1e-5 FT_EPOCHS=3 FT_SAVE_TOPK=1 \
  python reptorep_finetune_rattray.py
s2=$?
echo "=== two-stage done $(date) exit $s2 ==="
exit $s2
