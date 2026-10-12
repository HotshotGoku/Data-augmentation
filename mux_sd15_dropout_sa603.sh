#!/bin/bash
#SBATCH --job-name=mux_sd15_drop
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Fill matrix cell: conditioning-dropout SD1.5 ControlNet on MULTIPLEXED. train -> gen -> score (one job).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
MPX=/hpc/group/youlab/sa603/data/multiplexed_eval
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# adapter: mux manifest (source_path/target_path) -> cldm {source,target,prompt,folder} jsonl
SRC=$MPX/train_multiplexed.json
DST=$MPX/train_multiplexed_cldm.jsonl
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
export FT_LR=${FT_LR:-1e-5} FT_EPOCHS=${FT_EPOCHS:-40} FT_FREEZE=${FT_FREEZE:-none}
export FT_SAVE_TOPK=1 FT_TAG=mux_sd15_dropout FT_SEED=42
export HINT_DROPOUT=${HINT_DROPOUT:-0.2}
echo "=== train mux dropout SD1.5 $(date) tag=$FT_TAG epochs=$FT_EPOCHS dropout=$HINT_DROPOUT ==="
python reptorep_finetune.py
echo "=== train done $(date) exit=$? ==="
CKPT=$(ls -t finetune_runs/mux_sd15_dropout/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -f "$CKPT" ] || { echo "FATAL: no ckpt"; exit 1; }
echo "ckpt=$CKPT"

GEN=$REPO/mux_dropout_eval/gen; mkdir -p "$GEN"
echo "=== gen on mux sources $(date) ==="
FT_EVAL_CKPT="$CKPT" python reptorep_infer_rattray.py --input_dir "$MPX/sources" --out_dir "$GEN"

echo "=== score vs mux reals_heldout $(date) ==="
python eval_metrics.py --gen_dir "$GEN" --real_spec "MPLEX:$MPX/reals_heldout" --out "$GEN/metrics" || echo "score note"
python - "$GEN/metrics/metrics.json" <<'PY'
import json,sys
o=json.load(open(sys.argv[1])).get("overall",{}); g=lambda k:(o[k]["mean"] if isinstance(o.get(k),dict) else o.get(k))
print(f">>> SD1.5 DROPOUT (mux): realism={g('realism_vs_sibling'):.3f} CMMD={g('CMMD'):.2f} diversity={g('diversity'):.3f}")
print(">>> REF mux: SD1.5 in-domain CMMD 12.6 | PixCell 5.78")
PY
echo "=== DONE $(date) ==="
