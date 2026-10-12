#!/bin/bash
#SBATCH --job-name=sd15_branch_train
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Check #1: a FAIR SD1.5 baseline. Train a fresh SD1.5 ControlNet from the stock control_sd15 seed on
# EXACTLY train_branching.json (same split/exclusions as the Flux CN) -> apples-to-apples image metrics.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# cldm MyDataset wants {source,target,prompt,folder} JSONL; the Flux manifest has source_path/target_path.
SRC=/hpc/group/youlab/sa603/data/branching_256/train_branching.json
DST=/hpc/group/youlab/sa603/data/branching_256/train_branching_cldm.jsonl
python - "$SRC" "$DST" <<'PY'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]; n = 0
with open(src) as f, open(dst, "w") as o:
    for line in f:
        line = line.strip()
        if not line: continue
        r = json.loads(line); sp, tp = r["source_path"], r["target_path"]
        o.write(json.dumps({"source": os.path.basename(sp), "target": os.path.basename(tp),
                            "prompt": r.get("prompt", ""), "folder": os.path.dirname(sp)}) + "\n"); n += 1
print(f"[adapter] wrote {n} pairs -> {dst}")
PY

export FT_JSON="$DST"
export FT_RESUME="$SIM/sim_to_exp_diffusion/controlnet_essential/control_sd15_ini.ckpt"
export FT_LR=${FT_LR:-1e-5} FT_EPOCHS=${FT_EPOCHS:-150} FT_FREEZE=${FT_FREEZE:-none}
export FT_SAVE_TOPK=${FT_SAVE_TOPK:-1} FT_TAG=${FT_TAG:-branching_sd15_fromseed} FT_SEED=${FT_SEED:-42}  # topk must be 1 or -1 w/o a monitor metric
echo "=== SD1.5 branching train from seed $(date) tag=$FT_TAG epochs=$FT_EPOCHS lr=$FT_LR ==="
python reptorep_finetune.py
echo "=== done $(date) exit=$? ==="; ls -la finetune_runs/$FT_TAG/lightning_logs/version_*/checkpoints/ 2>/dev/null | tail
