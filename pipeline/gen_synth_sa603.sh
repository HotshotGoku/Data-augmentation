#!/bin/bash
#SBATCH --job-name=gen_synth
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Generate synthetic multiplexed replicate sets for (a) the downstream re-run and (b) Kinshuk handoff.
# Two models -> two folders: synth_ft (multiplexed-ft) and synth_generalist (generalist).
# Sources = real reps 8,9,10 (present in all 70 conditions), SYNTH_PER_SRC each.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

MPLEX_FT=$REPO/rattray_runs/multiplexed_ft/lightning_logs/version_54549115/checkpoints/epoch=2-step=1448.ckpt
GEN_CKPT=$REPO/rattray_runs/generalist/lightning_logs/version_54592625/checkpoints/epoch=3-step=5723.ckpt
PER=${SYNTH_PER_SRC:-8}

echo "=== synth_ft (multiplexed-ft) $(date) ==="
SYNTH_CKPT="$MPLEX_FT" SYNTH_DIR="$MPX/synth_ft" SYNTH_PER_SRC="$PER" python gen_multiplexed_synth.py
echo "=== synth_generalist (generalist) $(date) ==="
SYNTH_CKPT="$GEN_CKPT" SYNTH_DIR="$MPX/synth_generalist" SYNTH_PER_SRC="$PER" python gen_multiplexed_synth.py

echo "=== counts ==="
for d in synth_ft synth_generalist; do printf "%s: " "$d"; ls "$MPX/$d"/*.png 2>/dev/null | wc -l; done
echo "=== DONE $(date) ==="
