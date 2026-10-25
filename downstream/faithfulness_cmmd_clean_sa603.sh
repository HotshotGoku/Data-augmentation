#!/bin/bash
#SBATCH --job-name=cmmd_clean
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err

# P3 realism descriptor (SECONDARY to the probe): CMMD(synth vs clean reals) for all 8 clean configs.
# sbatch [--export=ALL,CAP=1000] downstream/faithfulness_cmmd_clean_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
cd "$REPO"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export PYTHONPATH="$REPO/pipeline/metrics:$REPO:${PYTHONPATH:-}"   # for `import cmmd`
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
export OUT_TSV=$REPO/faithfulness_out/cmmd_clean.tsv
echo "=== CMMD clean (8 configs) $(date) ==="
python downstream/faithfulness_cmmd_clean.py
echo "=== DONE $(date) ==="
column -t -s$'\t' "$OUT_TSV" 2>/dev/null || cat "$OUT_TSV"
