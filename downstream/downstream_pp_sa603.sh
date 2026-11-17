#!/bin/bash
#SBATCH --job-name=ds_pp
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# 15-seed downstream aug-vs-rotation on the NEW pp preprocessing domain. Single augmenter arm =
# the <ds>_sd15_pp ckpt's synth (synth_pp). Reuses the validated P5 harnesses verbatim: rotation arm
# (classical) = same reals per seed as the augmenter arm (apples-to-apples); GRAY channel-matches
# brightfield to kill the SD1.5 color-tint confound (mux keeps color = label). resnet50 standardized.
# Arms: real / real_dup / rotation / augmenter_pp / real_ceiling.
# Submit: sbatch --export=ALL,DS=<branching|mux|rattray> downstream/downstream_pp_sa603.sh
set -uo pipefail
REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
D=/hpc/group/youlab/sa603/data
source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"; cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"
export HF_HOME=/hpc/group/youlab/sa603/.hf_home
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'FATAL: no CUDA'; exit 1; }
DS=${DS:?set DS}; echo "=== ds_pp DS=$DS $(date) ==="

case "$DS" in
 branching)  # colony-feature regression (shape_factor headline + branch-arch), GRAY, N in {20,60}
   ODIR=$REPO/downstream/outputs/branching_downstream_out; mkdir -p "$ODIR"
   export TRAIN_REAL_DIR=$D/branching_256_pp/canon TEST_SRC_DIR=$D/branching_256_pp/test/sources TEST_REAL_DIR=$D/branching_256_pp/test/reals
   SYN=$D/branching_256_pp/synth_train_pp; FCSV=$ODIR/features_branching_pp.csv; RESULTS=$ODIR/results_pp.tsv
   done_already () { [ -f "$RESULTS" ] && awk -F'\t' -v t="$1" -v n="$2" -v s="$3" '$1==t&&$2==n&&$3==s{f=1} END{exit !f}' "$RESULTS"; }
   run () { # tag mode nt seed
     if done_already "$1" "$3" "$4"; then echo "skip $1 N=$3 s=$4"; return; fi
     echo "=== $1 N=$3 seed=$4 $(date +%H:%M:%S) ==="
     MODE="$2" MODE_TAG="$1" SYNTH_DIR="$SYN" FEATURES_CSV="$FCSV" N_TRAIN="$3" SEED="$4" \
       BACKBONE=resnet50 EPOCHS=50 N_PER=8 NW=8 GRAY=1 RESULTS_TSV="$RESULTS" \
       python downstream/downstream_colony_features.py || echo "  FAILED $1 N=$3 s=$4"; }
   for SEED in $(seq 0 14); do
     for NT in 20 60; do
       run real real "$NT" "$SEED"; run real_dup real_dup "$NT" "$SEED"
       run rotation classical "$NT" "$SEED"; run augmenter_pp augmenter "$NT" "$SEED"
     done
     run real_ceiling real all "$SEED"
   done ;;
 mux)        # aTc(row)/IPTG(col) decode, COLOR (=label, no GRAY), K in {1,2,3}
   ODIR=$REPO/downstream/outputs/downstream_multiplexed_out; mkdir -p "$ODIR"
   REALS=$D/multiplexed_eval/reals; SYN=$D/multiplexed_eval/synth_pp; RESULTS=$ODIR/results_pp.tsv
   done_already () { [ -f "$RESULTS" ] && awk -F'\t' -v t="$1" -v k="$2" -v s="$3" '$1==t&&$2==k&&$3==s{f=1} END{exit !f}' "$RESULTS"; }
   run () { # tag mode k seed
     if done_already "$1" "$3" "$4"; then echo "skip $1 K=$3 s=$4"; return; fi
     echo "=== $1 K=$3 seed=$4 $(date +%H:%M:%S) ==="
     MODE="$2" MODE_TAG="$1" SYNTH_DIR="$SYN" K="$3" SEED="$4" \
       BACKBONE=resnet50 EPOCHS=40 N_PER=8 REALS_DIR="$REALS" RESULTS_TSV="$RESULTS" \
       python downstream/downstream_multiplexed_decode.py || echo "  FAILED $1 K=$3 s=$4"; }
   for SEED in $(seq 0 14); do
     for K in 1 2 3; do
       run real real "$K" "$SEED"; run real_dup real_dup "$K" "$SEED"
       run rotation classical "$K" "$SEED"; run augmenter_pp augmenter "$K" "$SEED"
     done
     run real_ceiling real all "$SEED"
   done ;;
 rattray)    # 62-way strain classification, GRAY channel-match, held-out test rep = 1
   export REALS="$D/external_datasets/rattray_2023/processed_pp/*.png" \
          LABEL_RE='PIL(?P<lab>[0-9]+)-(?P<rep>[0-9]+)' KIND=cls LOGRATIO=0 TEST_REP=1 GRAY=1
   SYN=$D/external_datasets/rattray_2023/synth_pp
   OUT="$REPO/downstream/outputs/rattray_downstream_pp.tsv"; mkdir -p "$(dirname "$OUT")"; rm -f "$OUT"
   export K=1 N_PER=8 SEEDS=${SEEDS:-15} BACKBONE=resnet50 EPOCHS=${EPOCHS:-30} OUT
   MODE=real      TAG=rattray            python datasets/harness/ds_downstream.py
   MODE=real_dup  TAG=rattray            python datasets/harness/ds_downstream.py
   MODE=rotation  TAG=rattray            python datasets/harness/ds_downstream.py
   MODE=augmenter TAG=rattray_pp SYNTH="$SYN" python datasets/harness/ds_downstream.py
   MODE=real      TAG=rattray_ceiling K=999 python datasets/harness/ds_downstream.py
   echo "=== rattray SUMMARY ==="; python datasets/harness/ds_analyze.py "$OUT" ;;
 *) echo "unknown DS=$DS"; exit 1 ;;
esac
echo "=== DONE $DS $(date) ==="
