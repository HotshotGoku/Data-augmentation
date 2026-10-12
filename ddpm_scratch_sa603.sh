#!/bin/bash
#SBATCH --job-name=ddpm_scratch
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
ENV=/hpc/group/youlab/sa603/envs/ddpm
PNG=/hpc/group/youlab/sa603/data/branching_256/canon_png
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -p "$ENV" python=3.10; conda activate "$ENV"
  pip install -q torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -q denoising-diffusion-pytorch || { echo FATAL pip; exit 1; }
else conda activate "$ENV"; fi
cd "$REPO"
echo "=== DDPM-from-scratch train+sample (402 branching imgs) $(date) ==="
python - <<PY
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch, os
model = Unet(dim=64, dim_mults=(1,2,4,8), flash_attn=False)
diff = GaussianDiffusion(model, image_size=256, timesteps=1000)
tr = Trainer(diff, "$PNG", train_batch_size=8, train_lr=8e-5, train_num_steps=${STEPS:-40000},
             gradient_accumulate_every=2, ema_decay=0.995, amp=True, calculate_fid=False,
             results_folder="$REPO/ddpm_runs", save_and_sample_every=10000, num_samples=4)
tr.train()
# generate 400 individual images for scoring
import torchvision
out="$REPO/ddpm_runs/gen"; os.makedirs(out, exist_ok=True)
diff.eval(); n=0
with torch.no_grad():
    for _ in range(25):
        imgs = tr.ema.ema_model.sample(batch_size=16)
        for j in range(imgs.shape[0]):
            torchvision.utils.save_image(imgs[j], f"{out}/gen_{n:04d}.png"); n+=1
print("GENERATED", n, "->", out)
PY
echo "=== score (separate env) $(date) ==="
conda deactivate; conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
G=$REPO/ddpm_runs/gen python cmmd_direct.py 2>&1 | grep -E "CMMD|diversity|REFERENCE"
echo "=== DONE $(date) ==="
