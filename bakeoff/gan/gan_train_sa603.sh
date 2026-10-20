#!/bin/bash
#SBATCH --job-name=lwgan
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Low-data GAN track (the #1 literature signal). lightweight-gan (FastGAN): pip-installable, folder
# input, built for hundreds of 256px images. Self-contained: makes its env once, converts TIF->PNG,
# trains unconditionally on the 402 branching images (scored later on CMMD+diversity vs test/reals).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
ENV=/hpc/group/youlab/sa603/envs/lwgan
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# --- env (create once) ---
if [ ! -x "$ENV/bin/python" ]; then
  echo "=== creating lwgan env $(date) ==="
  conda create -y -p "$ENV" python=3.10
  conda activate "$ENV"
  pip install -q torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -q lightweight-gan || { echo "FATAL pip lightweight-gan"; exit 1; }
else
  conda activate "$ENV"
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- dataset: convert branching canon TIF -> PNG once ---
PNG=/hpc/group/youlab/sa603/data/branching_256/canon_png
if [ ! -d "$PNG" ] || [ "$(ls "$PNG"/*.png 2>/dev/null | wc -l)" -lt 300 ]; then
  echo "=== converting TIF->PNG ==="
  python - <<'PY'
import glob, os
from PIL import Image
src="/hpc/group/youlab/sa603/data/branching_256/canon"; dst=src+"_png"; os.makedirs(dst,exist_ok=True)
fs=glob.glob(src+"/*.TIF"); n=0
for f in fs:
    Image.open(f).convert("RGB").resize((256,256)).save(os.path.join(dst,os.path.basename(f)[:-4]+".png")); n+=1
print(f"converted {n} -> {dst}")
PY
fi

cd "$REPO"
echo "=== lwgan train start $(date) steps=${STEPS:-60000} ==="
lightweight_gan --data "$PNG" --image-size 256 --batch-size ${BS:-16} \
  --num-train-steps ${STEPS:-60000} --save-every 10000 --evaluate-every 100000 \
  --name lwgan_branching --results-dir "$REPO/gan_runs/results" --models-dir "$REPO/gan_runs/models" \
  --aug-prob 0.5 --disable-cuda-reproducibility
echo "=== train done $(date) exit=$? ==="
