#!/bin/bash
#SBATCH --job-name=gen_branch_synth
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Synthetic branching replicates for the colony-feature downstream experiment (generalist augmenter).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

GEN_CKPT=$REPO/rattray_runs/generalist/lightning_logs/version_54592625/checkpoints/epoch=3-step=5723.ckpt
OUT=/hpc/group/youlab/sa603/data/branching_downstream/synth_train_generalist
PER=${SYNTH_PER_SRC:-8}

echo "=== gen branching TRAIN-POOL synth (generalist) $(date) ckpt=$GEN_CKPT per=$PER ==="
SYNTH_CKPT="$GEN_CKPT" SYNTH_DIR="$OUT" \
  REAL_DIR=/hpc/group/youlab/sa603/data/branching_256/canon \
  EXCLUDE_DIR=/hpc/group/youlab/sa603/data/branching_256/test/reals \
  SYNTH_PER_SRC="$PER" python gen_branching_synth.py
echo "=== count: $(ls "$OUT"/*.png 2>/dev/null | wc -l) synth -> $OUT ==="

echo "=== LABEL canon + test + synth -> features_branching_all.csv $(date) ==="
D=/hpc/group/youlab/sa603/data; ODIR=$REPO/branching_downstream_out; mkdir -p "$ODIR"
python colony_features.py --images "$D/branching_256/canon/*.TIF"      --out /tmp/f_canon.csv || echo "label canon FAILED"
python colony_features.py --images "$D/branching_256/test/sources/*.TIF" --out /tmp/f_tsrc.csv || echo "label tsrc FAILED"
python colony_features.py --images "$D/branching_256/test/reals/*.TIF"   --out /tmp/f_treal.csv || echo "label treal FAILED"
python colony_features.py --images "$OUT/*.png"                          --out /tmp/f_synth.csv || echo "label synth FAILED"
head -1 /tmp/f_canon.csv > "$ODIR/features_branching_all.csv"
tail -q -n +2 /tmp/f_canon.csv /tmp/f_tsrc.csv /tmp/f_treal.csv /tmp/f_synth.csv >> "$ODIR/features_branching_all.csv"
echo "=== labeled $(($(wc -l < "$ODIR/features_branching_all.csv")-1)) images -> $ODIR/features_branching_all.csv ==="
echo "=== DONE $(date) ==="
