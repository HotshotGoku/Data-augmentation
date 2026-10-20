#!/bin/bash
#SBATCH --job-name=reptorep_ood_sa603
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH -o /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.out
#SBATCH -e /hpc/group/youlab/sa603/code/Data_augmentation/slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa603@duke.edu

# OUT-OF-DISTRIBUTION test: run the augmenter on a held-out SPECIES (KL two-species
# fluorescence co-culture, staged at data/ood_kl_2species), then score it with the same
# metrics. Compare eval_metrics_results_ood/ against the in-domain eval_metrics_results/
# to quantify the generalization gap. One job does inference THEN eval.
#
# Submit with:  sbatch Slurm_scripts/ood_test_sa603.sh

set -uo pipefail

REPO=/hpc/group/youlab/sa603/code/Data_augmentation
SIM=/hpc/group/youlab/sa603/code/Simulation_templated_pattern_prediction
OOD_SRC=/hpc/group/youlab/sa603/data/ood_kl_2species
OOD_GEN=$REPO/inference_ood_kl

source /hpc/home/sa603/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_PA_patternprediction
export PHYSICS_DL_PROJECT_PATH="$SIM"
cd "$REPO"
export PYTHONPATH="$REPO:$(dirname "$REPO"):$REPO/utils:${PYTHONPATH:-}"

echo "=== start $(date) | host $(hostname) | job ${SLURM_JOB_ID:-?} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo 'FATAL: CUDA not available — aborting.'; exit 1; }

echo "--- 1/2 OOD inference ---"
python reptorep_infer_ood.py --input_dir "$OOD_SRC" --out_dir "$OOD_GEN"
istat=$?
echo "--- 2/2 OOD metrics ---"
python eval_metrics.py --gen_dir "$OOD_GEN" --real_spec "KL2SP:$OOD_SRC" --out eval_metrics_results_ood
estat=$?

echo "=== done $(date) | infer exit $istat | eval exit $estat ==="
exit $(( istat || estat ))
