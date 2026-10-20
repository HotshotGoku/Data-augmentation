#!/bin/bash
#SBATCH --job-name=vae_diag
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Phase 4a: download 4ch (SD1.5) + 16ch (ostris, and SD3.5 if licensed) VAEs, run reconstruction diagnostic.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
ENVDIR=/hpc/group/youlab/sa603/envs/flowmatch
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
VAEROOT=/hpc/group/youlab/sa603/models/vaes
mkdir -p "$VAEROOT"
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVDIR"
cd "$REPO"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

dl () {  # repo subfolder destdir
  python - "$1" "$2" "$3" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
repo, sub, dest = sys.argv[1], sys.argv[2], sys.argv[3]
pats = [f"{sub}/*"] if sub else ["*.json", "*.safetensors"]
try:
    snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=pats, token=os.environ.get("HF_TOKEN"))
    print("  DL OK:", repo, sub or "")
except Exception as e:
    print("  DL FAIL:", repo, sub or "", "->", str(e)[:200])
PY
}
echo "=== downloading VAEs $(date) ==="
dl stabilityai/sd-vae-ft-mse "" "$VAEROOT/sd15_vae"
dl ostris/vae-kl-f8-d16 "" "$VAEROOT/ostris16"
dl stabilityai/stable-diffusion-3.5-medium "vae" "$VAEROOT/sd35"   # gated: only works if license accepted

export SD15_VAE="$VAEROOT/sd15_vae"
export VAE16_A="$VAEROOT/ostris16"
[ -f "$VAEROOT/sd35/vae/config.json" ] && export VAE16_B="$VAEROOT/sd35/vae" && echo "SD3.5 VAE present (licensed)"

echo "=== running reconstruction diagnostic $(date) ==="
python reconstruct_vae_test.py
echo "=== vae_diag done $(date) ==="
