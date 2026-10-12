#!/bin/bash
#SBATCH --job-name=sdxl_cn
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# SDXL ControlNet (official diffusers script). Strong general UNet prior, conditioned (comparable to
# SD1.5/Flux). Reuses the SD3 imagefolder dataset. Compute node has internet -> fetches fp16-VAE fix.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
export HF_TOKEN=$(cat "$HOME/.hf_token")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/flowmatch
pip install -q "datasets==2.21.0" "huggingface_hub==0.25.2" 2>&1 | tail -2   # datasets for imagefolder; pin hf-hub<1.0 (diffusers0.31 needs it)
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
[ -f train_controlnet_sdxl.py ] || curl -sL https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/examples/controlnet/train_controlnet_sdxl.py -o train_controlnet_sdxl.py
# shim load_dataset -> load_from_disk for our 2-Image-column saved dataset
python - train_controlnet_sdxl.py <<'PY'
import sys; f=sys.argv[1]; s=open(f).read()
shim=("from datasets import load_dataset as _ld, load_from_disk as _lfd\nimport os as _os\n"
      "def load_dataset(p,*a,**k):\n    return _lfd(p) if (isinstance(p,str) and _os.path.isdir(p) and _os.path.exists(_os.path.join(p,'dataset_dict.json'))) else _ld(p,*a,**k)\n")
if "_lfd" not in s: open(f,"w").write(s.replace("from datasets import load_dataset\n",shim,1)); print("patched",f)
else: print("already patched")
PY
DATA=/hpc/group/youlab/sa603/data/branching_256/sd3_cn_ds
[ -d "$DATA" ] || python build_sd3_dataset.py
echo "=== SDXL CN ${RUN:-smoke} start $(date) steps=${STEPS:-10} ==="
accelerate launch --num_processes=1 --mixed_precision=bf16 train_controlnet_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --mixed_precision=bf16 \
  --train_data_dir="$DATA" --image_column=image --conditioning_image_column=conditioning_image --caption_column=text \
  --output_dir="$REPO/sdxl_cn_runs/${RUN:-smoke}" --resolution=256 --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing --use_8bit_adam --learning_rate=1e-5 \
  --max_train_steps=${STEPS:-10} --checkpointing_steps=${CKPT:-5000} --seed=42 --dataloader_num_workers=4
echo "=== done $(date) exit=$? ==="
