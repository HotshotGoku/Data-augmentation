#!/bin/bash
#SBATCH --job-name=flux_smoke
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Track B: Flux ControlNet feasibility smoke on a 24GB A5000. Needs HF_TOKEN with FLUX.1-dev
# license accepted (pass via --export=ALL,HF_TOKEN=...). Downloads Flux (~24G) + T5 + ControlNet.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
ENVDIR=/hpc/group/youlab/sa603/envs/flowmatch
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVDIR"
cd "$REPO"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "=== flux smoke start $(date) (offload=${OFFLOAD:-sequential} HW=${HW:-512}) ==="
OFFLOAD=${OFFLOAD:-sequential} HW=${HW:-512} STEPS=${STEPS:-24} python flux_controlnet_smoke.py
status=$?
echo "=== flux smoke done $(date) exit $status ==="
exit $status
