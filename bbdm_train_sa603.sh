#!/bin/bash
#SBATCH --job-name=bbdm_train
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Train Latent BBDM (LBBDM-f4) on the multiplexed replicate pairs (source->target bridge).
# Config generated from the template (eta=1.0 max diversity, condition_key=nocond kept as defaults).
# Smoke first with MAX_STEPS=300 to verify data pairing (sample_at_start grid) + that it trains;
# then full run with MAX_STEPS unset.
set -uo pipefail
BBDM=/hpc/group/youlab/sa603/code/BBDM
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/youlab/sa603/envs/bbdm
cd "$BBDM"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }

CFG=configs/youlab-LBBDM-f4.yaml
cp configs/Template-LBBDM-f4.yaml "$CFG"
sed -i "s#dataset_name: 'dataset_name'#dataset_name: '${DS_NAME:-mplex}'#" "$CFG"
sed -i "s#dataset_path: 'dataset_path'#dataset_path: '/hpc/group/youlab/sa603/data/bbdm_multiplexed'#" "$CFG"
sed -i "s#ckpt_path: 'results/VQGAN/CelebAMaskHQ-f4.ckpt'#ckpt_path: '/hpc/group/youlab/sa603/code/BBDM/results/VQGAN/model.ckpt'#" "$CFG"
sed -i "s#start_ema_step: 30000#start_ema_step: 3000#" "$CFG"
sed -i "s#sample_num: 5#sample_num: 8#" "$CFG"
sed -i "s#max_var: 1.0#max_var: ${MAX_VAR:-1.0}#" "$CFG"   # >1.0 widens the bridge = more diversity
echo "=== config changes vs template ==="; diff configs/Template-LBBDM-f4.yaml "$CFG" || true

EXTRA=""; [ -n "${MAX_STEPS:-}" ] && EXTRA="--max_steps ${MAX_STEPS}"
echo "=== BBDM train start $(date) MAX_STEPS=${MAX_STEPS:-full} ==="
python3 main.py --config "$CFG" -r results --train --sample_at_start --save_top --gpu_ids 0 $EXTRA
echo "=== BBDM train exit=$? $(date) ==="
echo "=== ckpts + latest sample grids ==="
find "$BBDM/results" -name "*.pth" 2>/dev/null | tail -10
find "$BBDM/results" -name "*.png" 2>/dev/null | tail -10
