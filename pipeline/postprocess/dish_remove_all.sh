#!/bin/bash
# Phase 0: apply dish removal (rim 0.20, user-approved 2026-10-21) UNIFORMLY to every real dataset ->
# drop-in clean mirror under data/clean/<same relative path> (same filenames+extensions so harnesses
# just repoint their dir env vars). Originals kept untouched. Processing is standardized to 256 (the
# whole pipeline's working size). CPU; run on dcc-agent or sbatch a CPU partition.
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
DATA=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
cd "$REPO"
RIM=${RIM:-0.20}

DIRS=(
  branching_256/canon
  branching_256/test/sources
  branching_256/test/reals
  multiplexed_eval/reals
  multiplexed_eval/reals_heldout
  multiplexed_eval/sources
  ood_kl_2species
  external_datasets/rattray_2023/images
  external_datasets/rattray_2023/processed
  external_datasets/rattray_2023/test_reals
  external_datasets/rattray_2023/test_sources
)

for d in "${DIRS[@]}"; do
  src="$DATA/$d"; dst="$DATA/clean/$d"
  echo "=== $d ($(ls "$src" 2>/dev/null | wc -l) src files) $(date +%H:%M:%S) ==="
  python pipeline/postprocess/dish_remove.py "$src" "$dst" --rim-frac "$RIM"
done

echo "=== count check (src vs clean) ==="
for d in "${DIRS[@]}"; do
  printf "%-45s src=%s clean=%s\n" "$d" "$(ls "$DATA/$d" 2>/dev/null | wc -l)" "$(ls "$DATA/clean/$d" 2>/dev/null | wc -l)"
done
echo "DONE $(date)"
