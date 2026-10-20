#!/bin/bash
#SBATCH --job-name=rattray_pilot
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Phase-1 pilot: rattray end-to-end. gen (in-domain + dropout) -> strain-ID downstream (real/rotation/
# augmenter-indomain/augmenter-dropout, 5 seeds) -> paired summary. Validates every harness before fan-out.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
RAT=/hpc/group/youlab/sa603/data/external_datasets/rattray_2023
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

CK_IN=$(ls -t "$REPO"/finetune_runs/rattray_sd15/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
CK_DR=$(ls -t "$REPO"/finetune_runs/rattray_sd15_dropout/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
echo "in-domain ckpt=$CK_IN"; echo "dropout ckpt=$CK_DR"

echo "=== GEN in-domain $(date) ==="
FT_EVAL_CKPT="$CK_IN" SOURCES="$RAT/processed/*.png" OUT="$REPO/rattray_synth_indomain" N=8 python ds_gen.py
echo "=== GEN dropout $(date) ==="
FT_EVAL_CKPT="$CK_DR" SOURCES="$RAT/processed/*.png" OUT="$REPO/rattray_synth_dropout" N=8 python ds_gen.py

echo "=== DOWNSTREAM strain-ID (4 arms x 5 seeds) $(date) ==="
export REALS="$RAT/processed/*.png"
export LABEL_RE='PIL(?P<lab>[0-9]+)-(?P<rep>[0-9]+)'
export KIND=cls TEST_REP=1 K=1 N_PER=8 SEEDS=5 OUT="$REPO/rattray_downstream.tsv"
rm -f "$OUT"
MODE=real      TAG=rattray          python ds_downstream.py
MODE=rotation  TAG=rattray          python ds_downstream.py
MODE=augmenter TAG=rattray_indomain SYNTH="$REPO/rattray_synth_indomain" python ds_downstream.py
MODE=augmenter TAG=rattray_dropout  SYNTH="$REPO/rattray_synth_dropout"  python ds_downstream.py

echo "=== PILOT SUMMARY $(date) ==="
python ds_analyze.py "$OUT"
echo "=== DONE $(date) ==="
