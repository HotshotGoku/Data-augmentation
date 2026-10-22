#!/bin/bash
#SBATCH --job-name=ds_sd15_train
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Generic in-domain SD1.5 ControlNet trainer for ANY dataset (reuses the branching recipe).
# Submit: sbatch --export=ALL,MANIFEST=<path.json>,FT_TAG=<tag>[,HINT_DROPOUT=0.2][,FT_EPOCHS=150] ds_sd15_train_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$REPO/pipeline:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"  # pipeline/ on path post-reorg (reptorep_* moved there)
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

MANIFEST=${MANIFEST:?set MANIFEST}
DST=${MANIFEST%.json}_cldm.jsonl
python - "$MANIFEST" "$DST" <<'PY'
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
export FT_SAVE_TOPK=1 FT_TAG=${FT_TAG:?set FT_TAG} FT_SEED=42
export HINT_DROPOUT=${HINT_DROPOUT:-0.0}
echo "=== SD1.5 train MANIFEST=$MANIFEST tag=$FT_TAG epochs=$FT_EPOCHS dropout=$HINT_DROPOUT $(date) ==="
python pipeline/reptorep_finetune.py; rc=$?
echo "=== done $(date) exit=$rc ==="; ls -la finetune_runs/$FT_TAG/lightning_logs/version_*/checkpoints/ 2>/dev/null | tail
