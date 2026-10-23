#!/bin/bash
#SBATCH --job-name=gen_clean
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Gen synth for mux|kl|rattray from a CLEAN augmenter ckpt via ds_gen.py, with the post-Oct-20-reorg
# paths fixed (pipeline/ on PYTHONPATH so `import pipeline`->pipeline.py; utils->pipeline/utils symlink).
# mux loops train reps 8,9,10 (one ds_gen glob per rep, accumulating into one OUT) to mirror
# gen_multiplexed_synth.py's source selection without editing it. Names <stem>_s<i>.png (label in stem).
# Submit: sbatch --export=ALL,DS=mux|kl|rattray,SYNTH_CKPT=<ckpt>,TAG=clean|clean_dropout datasets/harness/gen_clean_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$REPO/pipeline:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"  # pipeline/ for import pipeline; utils symlink for Data_augmentation.utils
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

DS=${DS:?set DS}; CK=${SYNTH_CKPT:?set SYNTH_CKPT}; TAG=${TAG:?set TAG}; PER=${SYNTH_PER_SRC:-8}
DK=/hpc/group/youlab/sa603/data/clean
echo "=== gen clean DS=$DS tag=$TAG $(date) ckpt=$CK ==="
case "$DS" in
  mux)     OUT=$DK/multiplexed_eval/synth_$TAG; mkdir -p "$OUT"
           for rep in 8 9 10; do
             echo "--- mux rep$rep ---"
             FT_EVAL_CKPT="$CK" SOURCES="$DK/multiplexed_eval/reals/*_Rep${rep}.TIF" OUT="$OUT" N="$PER" python datasets/harness/ds_gen.py
           done ;;
  kl)      OUT=$DK/ood_kl_2species/synth_$TAG; mkdir -p "$OUT"
           FT_EVAL_CKPT="$CK" SOURCES="$DK/ood_kl_2species/*.TIF" OUT="$OUT" N="$PER" python datasets/harness/ds_gen.py ;;
  rattray) OUT=$DK/external_datasets/rattray_2023/synth_$TAG; mkdir -p "$OUT"
           FT_EVAL_CKPT="$CK" SOURCES="$DK/external_datasets/rattray_2023/processed/*.png" OUT="$OUT" N="$PER" python datasets/harness/ds_gen.py ;;
  *) echo "unknown DS=$DS"; exit 1 ;;
esac
echo "=== gen done $(date) count=$(ls "$OUT"/*.png 2>/dev/null | wc -l) -> $OUT ==="
