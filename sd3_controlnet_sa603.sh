#!/bin/bash
#SBATCH --job-name=sd3_cn
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# SD3.5-medium ControlNet (official diffusers script + 1-line patch for a SMALL controlnet so it fits).
# #2 candidate: flow-matching WITH working CFG, conditioned (source-anchored) like SD1.5/Flux.
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

# dataset (build once)
DATA=/hpc/group/youlab/sa603/data/branching_256/sd3_cn_data
[ -f "$DATA/metadata.jsonl" ] || python build_sd3_dataset.py

# patch: small controlnet (num_layers=2) so SD3-medium + T5 + CN fit on 48GB
grep -q "num_layers=2" train_controlnet_sd3.py || \
  sed -i 's/SD3ControlNetModel.from_transformer(transformer)/SD3ControlNetModel.from_transformer(transformer, num_layers=2)/' train_controlnet_sd3.py
echo "patch check:"; grep -n "from_transformer(transformer" train_controlnet_sd3.py | head -1

echo "=== SD3 ControlNet ${RUN:-smoke} start $(date) steps=${STEPS:-10} ==="
accelerate launch --num_processes=1 --mixed_precision=fp16 train_controlnet_sd3.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium" \
  --train_data_dir="$DATA" --image_column=image --conditioning_image_column=conditioning_image --caption_column=text \
  --output_dir="$REPO/sd3_cn_runs/${RUN:-smoke}" --resolution=256 --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing --use_8bit_adam --learning_rate=1e-5 \
  --max_train_steps=${STEPS:-10} --checkpointing_steps=${CKPT:-5000} --seed=42 --dataloader_num_workers=4
echo "=== done $(date) exit=$? ==="
