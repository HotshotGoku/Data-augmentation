#!/bin/bash
#SBATCH --job-name=sg2ada
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Gold-standard low-data GAN (StyleGAN2-ADA, NVIDIA). Self-contained: dedicated env (torch 2.1.2 =
# best ADA custom-op compatibility), clone repo, pack 402 branching PNGs to a dataset zip, train with
# adaptive discriminator augmentation. The custom CUDA ops JIT-compile at train start (the risk point).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
ENV=/hpc/group/youlab/sa603/envs/sg2ada
SRC=/hpc/group/youlab/sa603/code/stylegan2-ada-pytorch
PNG=/hpc/group/youlab/sa603/data/branching_256/canon_png
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

if [ ! -x "$ENV/bin/python" ]; then
  echo "=== creating sg2ada env $(date) ==="
  conda create -y -p "$ENV" python=3.9
  conda activate "$ENV"
  pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
  pip install -q click requests tqdm pyspng ninja imageio imageio-ffmpeg==0.4.3 psutil scipy || { echo FATAL pip; exit 1; }
else
  conda activate "$ENV"
fi
pip install -q "numpy<2"   # ADA + torch2.1 need numpy 1.x (numpy2 breaks torch init + torch.full(np.float32) at networks.py:101)
python -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available())"

[ -d "$SRC" ] || git clone --depth 1 https://github.com/NVlabs/stylegan2-ada-pytorch "$SRC"
cd "$SRC"
# patch grid_sample custom-op version gate to allow torch 2.x (its manual double-backward supports R1 reg)
sed -i "s/for x in \['1\.7\.', '1\.8\.', '1\.9'\]/for x in ['1.','2.']/" torch_utils/ops/grid_sample_gradfix.py
grep -n "for x in \['1" torch_utils/ops/grid_sample_gradfix.py | head -1
DATA=$REPO/gan_runs/sg2ada_data.zip
[ -f "$DATA" ] || python dataset_tool.py --source="$PNG" --dest="$DATA"
echo "=== ADA train start (ops compile here) $(date) ==="
python train.py --outdir=$REPO/gan_runs/sg2ada_out --data="$DATA" --gpus=1 --cfg=paper256 \
  --aug=ada --kimg=${KIMG:-1000} --snap=${SNAP:-10} --metrics=none
echo "=== done $(date) exit=$? ==="
