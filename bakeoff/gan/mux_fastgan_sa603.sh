#!/bin/bash
#SBATCH --job-name=mux_lwgan
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Fill matrix cell: FastGAN (lightweight-gan) unconditional on MULTIPLEXED reals. Train only; gen+cmmd after.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
ENV=/hpc/group/youlab/sa603/envs/lwgan
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# dataset: mux reals TIF -> PNG once
PNG=$MPX/reals_png
if [ ! -d "$PNG" ] || [ "$(ls "$PNG"/*.png 2>/dev/null | wc -l)" -lt 300 ]; then
  echo "=== converting mux reals TIF->PNG ==="
  python - "$MPX/reals" "$PNG" <<'PY'
import glob, os, sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
n=0
for f in glob.glob(src+"/*.TIF"):
    Image.open(f).convert("RGB").resize((256,256)).save(os.path.join(dst, os.path.basename(f)[:-4]+".png")); n+=1
print(f"converted {n} -> {dst}")
PY
fi

cd "$REPO"
echo "=== lwgan mux train start $(date) steps=${STEPS:-60000} ==="
lightweight_gan --data "$PNG" --image-size 256 --batch-size ${BS:-16} \
  --num-train-steps ${STEPS:-60000} --save-every 10000 --evaluate-every 100000 \
  --name lwgan_mux --results-dir "$REPO/gan_runs/results" --models-dir "$REPO/gan_runs/models" \
  --aug-prob 0.5 --disable-cuda-reproducibility
echo "=== train done $(date) exit=$? ==="
