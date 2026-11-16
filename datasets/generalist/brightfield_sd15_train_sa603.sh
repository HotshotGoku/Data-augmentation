#!/bin/bash
#SBATCH --job-name=sd15_bf_train
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Standalone brightfield (pakp|nlev|selx) SD1.5 ControlNet from the stock seed, on the
# initialstage_mod-preprocessed _pp build + runtime rotate/flip/brightness (white fill). Set CODE:
#   sbatch --job-name=sd15_pakp --export=ALL,CODE=pakp datasets/generalist/brightfield_sd15_train_sa603.sh
set -uo pipefail
: "${CODE:?set CODE=pakp|nlev|selx}"
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# adapter: builder manifest (source_path/target_path) -> cldm {source,target,prompt,folder} jsonl
SRC=/hpc/group/youlab/sa603/data/${CODE}_256_pp/train_${CODE}.json
DST=/hpc/group/youlab/sa603/data/${CODE}_256_pp/train_${CODE}_cldm.jsonl
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
export FT_SAVE_TOPK=${FT_SAVE_TOPK:-1} FT_TAG=${FT_TAG:-${CODE}_sd15_fromseed} FT_SEED=${FT_SEED:-42}
# Runtime augmentation (Kinshuk 2026-11-17): brightfield -> white fill, brightness ON.
export GEN_AUG=${GEN_AUG:-1} AUG_FILL=${AUG_FILL:-255} AUG_BRIGHTNESS=${AUG_BRIGHTNESS:-1}
echo "=== SD1.5 $CODE train from seed $(date) tag=$FT_TAG epochs=$FT_EPOCHS lr=$FT_LR GEN_AUG=$GEN_AUG ==="
python pipeline/reptorep_finetune.py   # script lives in pipeline/ after the 2026-10-20 reorg
echo "=== done $(date) exit=$? ==="; ls -la finetune_runs/$FT_TAG/lightning_logs/version_*/checkpoints/ 2>/dev/null | tail
