#!/bin/bash
#SBATCH --job-name=sd3_newenv
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
ENV=/hpc/group/youlab/sa603/envs/sd3cn
export HF_HOME=/hpc/group/youlab/sa603/.hf_home; export HF_TOKEN=$(cat "$HOME/.hf_token")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -p "$ENV" python=3.10; conda activate "$ENV"
  pip install -q torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -q "diffusers==0.32.2" "transformers==4.46.3" accelerate "datasets==2.21.0" "huggingface_hub<1.0" \
    bitsandbytes peft sentencepiece protobuf || { echo FATAL pip; exit 1; }
else conda activate "$ENV"; fi
cd "$REPO"
python -c "import diffusers; print('diffusers', diffusers.__version__)"
F=train_controlnet_sd3_v32.py
[ -f "$F" ] || curl -sL https://raw.githubusercontent.com/huggingface/diffusers/v0.32.2/examples/controlnet/train_controlnet_sd3.py -o "$F"
grep -q "num_layers=2" "$F" || sed -i 's/SD3ControlNetModel.from_transformer(transformer)/SD3ControlNetModel.from_transformer(transformer, num_layers=2)/' "$F"
python - "$F" <<'PY'
import sys; f=sys.argv[1]; s=open(f).read()
shim=("from datasets import load_dataset as _ld, load_from_disk as _lfd\nimport os as _os\n"
      "def load_dataset(p,*a,**k):\n    return _lfd(p) if (isinstance(p,str) and _os.path.isdir(p) and _os.path.exists(_os.path.join(p,'dataset_dict.json'))) else _ld(p,*a,**k)\n")
if "_lfd" not in s: open(f,"w").write(s.replace("from datasets import load_dataset\n",shim,1)); print("patched shim")
PY
echo "=== SD3.5 ControlNet (diffusers 0.32) ${RUN:-smoke} start $(date) steps=${STEPS:-10} ==="
accelerate launch --num_processes=1 --mixed_precision=bf16 "$F" \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium" \
  --train_data_dir=/hpc/group/youlab/sa603/data/branching_256/sd3_cn_ds \
  --image_column=image --conditioning_image_column=conditioning_image --caption_column=text \
  --output_dir="$REPO/sd3_cn_runs/${RUN:-smoke}" --resolution=256 --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing --use_8bit_adam --learning_rate=1e-5 \
  --max_train_steps=${STEPS:-10} --checkpointing_steps=${CKPT:-5000} --seed=42
echo "=== done $(date) exit=$? ==="
