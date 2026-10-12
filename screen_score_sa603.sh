#!/bin/bash
#SBATCH --job-name=bbscore
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# Score every screen output dir with the SAME eval_metrics.py (realism/CMMD/diversity vs branching
# test/reals), then print a ranked leaderboard. Separate clean job (no generator in-process).
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"; export HF_HOME=/hpc/group/youlab/sa603/.hf_home
REALS="FFBR:$D/branching_256/test/reals"

echo "=== scoring screen dirs $(date) ==="
for G in "$REPO"/backbone_screen/*/; do
  [ -z "$(ls "$G"/*.png 2>/dev/null)" ] && continue
  python eval_metrics.py --gen_dir "$G" --real_spec "$REALS" --out "$G/metrics" >/dev/null 2>&1 || echo "  score FAIL $G"
done

python - <<'PY'
import json, glob, os
rows=[]
for m in glob.glob("/hpc/group/youlab/sa603/code/Data_augmentation/backbone_screen/*/metrics/metrics.json"):
    try:
        o=json.load(open(m)).get("overall",{})
        g=lambda k:(o[k]["mean"] if isinstance(o.get(k),dict) else o.get(k))
        rows.append((os.path.basename(os.path.dirname(os.path.dirname(m))), g("realism_vs_sibling"), g("CMMD"), g("diversity")))
    except Exception as e: print("parse fail",m,e)
print("\n=== BACKBONE SCREEN LEADERBOARD (zero-shot img2img; lower real/CMMD better, higher div better) ===")
print(f"{'backbone_strength':26} {'realism':>8} {'CMMD':>8} {'diversity':>10}")
for n,r,c,d in sorted(rows, key=lambda x: (x[2] if x[2] else 1e9)):
    print(f"{n:26} {r:8.3f} {c:8.2f} {d:10.3f}")
print("\nREFERENCE (trained, in-domain): SD1.5 realism 0.324 / CMMD 3.38 / div 0.252 | Flux-bf16 0.503/4.84/0.210")
PY
echo "=== DONE $(date) ==="
