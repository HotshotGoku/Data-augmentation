#!/bin/bash
#SBATCH --job-name=kl_pilot
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Phase-2 KL: end-to-end. gen (in-domain + dropout) -> mixing-ratio REGRESSION downstream
# (real/rotation/aug-indomain/aug-dropout, 5 seeds) -> paired summary. Tests the continuous-axis prediction.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
KLD=/hpc/group/youlab/sa603/data/ood_kl_2species
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

CK_IN=$(ls -t "$REPO"/finetune_runs/kl_sd15/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
CK_DR=$(ls -t "$REPO"/finetune_runs/kl_sd15_dropout/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
echo "in-domain ckpt=$CK_IN"; echo "dropout ckpt=$CK_DR"

echo "=== GEN in-domain $(date) ==="
FT_EVAL_CKPT="$CK_IN" SOURCES="$KLD/*.TIF" OUT="$REPO/kl_synth_indomain" N=8 python ds_gen.py
echo "=== GEN dropout $(date) ==="
FT_EVAL_CKPT="$CK_DR" SOURCES="$KLD/*.TIF" OUT="$REPO/kl_synth_dropout" N=8 python ds_gen.py

echo "=== DOWNSTREAM ratio-regression (4 arms x 5 seeds) $(date) ==="
export REALS="$KLD/*.TIF"
export LABEL_RE='(?P<lab>[0-9]+-[0-9]+)_(?P<rep>[0-9]+)\.TIF'
export KIND=reg LOGRATIO=1 TEST_REP=1 K=1 N_PER=8 SEEDS=5 OUT="$REPO/kl_downstream.tsv"
rm -f "$OUT"
MODE=real      TAG=kl          python ds_downstream.py
MODE=rotation  TAG=kl          python ds_downstream.py
MODE=augmenter TAG=kl_indomain SYNTH="$REPO/kl_synth_indomain" python ds_downstream.py
MODE=augmenter TAG=kl_dropout  SYNTH="$REPO/kl_synth_dropout"  python ds_downstream.py

echo "=== KL SUMMARY $(date) ==="
python ds_analyze.py "$OUT"
echo "=== DONE $(date) ==="
