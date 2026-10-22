#!/bin/bash
#SBATCH --job-name=gen_branch_clean
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Gen branching TRAIN-POOL synth from a CLEAN (dish-removed) augmenter ckpt, then label
# canon+test+synth -> features CSV (incl. branch-arch cols). Clean-rerun, parametrized version
# of gen_branching_synth_sa603.sh with the post-Oct-20-reorg paths fixed (pipeline/ on PYTHONPATH,
# scripts under downstream/, utils->pipeline/utils symlink for Data_augmentation.utils imports).
# Submit: sbatch --export=ALL,SYNTH_CKPT=<ckpt>,SYNTH_TAG=clean[,SYNTH_PER_SRC=8] downstream/gen_branching_synth_clean_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$REPO/pipeline:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"  # pipeline/ for `import pipeline`(pipeline.py); utils symlink for Data_augmentation.utils
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

SYNTH_CKPT=${SYNTH_CKPT:?set SYNTH_CKPT}; TAG=${SYNTH_TAG:?set SYNTH_TAG}; PER=${SYNTH_PER_SRC:-8}
D=/hpc/group/youlab/sa603/data/clean/branching_256
OUT=/hpc/group/youlab/sa603/data/clean/branching_downstream/synth_train_${TAG}
ODIR=$REPO/downstream/outputs/branching_downstream_out; mkdir -p "$OUT" "$ODIR"

echo "=== gen branching clean synth $(date) tag=$TAG ckpt=$SYNTH_CKPT per=$PER ==="
SYNTH_CKPT="$SYNTH_CKPT" SYNTH_DIR="$OUT" REAL_DIR="$D/canon" EXCLUDE_DIR="$D/test/reals" \
  SYNTH_PER_SRC="$PER" python downstream/gen_branching_synth.py; rc=$?
echo "=== gen exit=$rc count=$(ls "$OUT"/*.png 2>/dev/null | wc -l) -> $OUT ==="

echo "=== LABEL clean canon+test+synth -> features_branching_${TAG}.csv $(date) ==="
python downstream/colony_features.py --images "$D/canon/*.TIF"        --out /tmp/f_canon_$TAG.csv || echo "label canon FAILED"
python downstream/colony_features.py --images "$D/test/sources/*.TIF" --out /tmp/f_tsrc_$TAG.csv  || echo "label tsrc FAILED"
python downstream/colony_features.py --images "$D/test/reals/*.TIF"   --out /tmp/f_treal_$TAG.csv || echo "label treal FAILED"
python downstream/colony_features.py --images "$OUT/*.png"            --out /tmp/f_synth_$TAG.csv || echo "label synth FAILED"
CSV=$ODIR/features_branching_${TAG}.csv
head -1 /tmp/f_canon_$TAG.csv > "$CSV"
tail -q -n +2 /tmp/f_canon_$TAG.csv /tmp/f_tsrc_$TAG.csv /tmp/f_treal_$TAG.csv /tmp/f_synth_$TAG.csv >> "$CSV"
echo "=== labeled $(($(wc -l < "$CSV")-1)) imgs -> $CSV $(date) ==="
