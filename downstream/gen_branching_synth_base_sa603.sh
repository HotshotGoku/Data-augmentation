#!/bin/bash
#SBATCH --job-name=gen_branch_base
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Robustness rerun: synth from the BASE (branching-specialized) augmenter, then label into a
# SEPARATE csv (base synth shares basenames with generalist synth, so must not collide).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

BASE_CKPT=$REPO/lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt
OUT=/hpc/group/youlab/sa603/data/branching_downstream/synth_train_base
PER=${SYNTH_PER_SRC:-8}
D=/hpc/group/youlab/sa603/data; ODIR=$REPO/branching_downstream_out

echo "=== gen branching TRAIN-POOL synth (BASE) $(date) ckpt=$BASE_CKPT per=$PER ==="
SYNTH_CKPT="$BASE_CKPT" SYNTH_DIR="$OUT" \
  REAL_DIR=$D/branching_256/canon EXCLUDE_DIR=$D/branching_256/test/reals \
  SYNTH_PER_SRC="$PER" python gen_branching_synth.py
echo "=== count: $(ls "$OUT"/*.png 2>/dev/null | wc -l) synth -> $OUT ==="

echo "=== LABEL canon + test + BASE synth -> features_branching_base.csv $(date) ==="
python colony_features.py --images "$D/branching_256/canon/*.TIF"       --out /tmp/fb_canon.csv || echo "label canon FAILED"
python colony_features.py --images "$D/branching_256/test/sources/*.TIF" --out /tmp/fb_tsrc.csv || echo "label tsrc FAILED"
python colony_features.py --images "$D/branching_256/test/reals/*.TIF"    --out /tmp/fb_treal.csv || echo "label treal FAILED"
python colony_features.py --images "$OUT/*.png"                           --out /tmp/fb_synth.csv || echo "label synth FAILED"
head -1 /tmp/fb_canon.csv > "$ODIR/features_branching_base.csv"
tail -q -n +2 /tmp/fb_canon.csv /tmp/fb_tsrc.csv /tmp/fb_treal.csv /tmp/fb_synth.csv >> "$ODIR/features_branching_base.csv"
echo "=== labeled $(($(wc -l < "$ODIR/features_branching_base.csv")-1)) images ==="
echo "=== DONE $(date) ==="
