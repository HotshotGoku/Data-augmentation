#!/bin/bash 
#SBATCH -o slurm_ExpAugmentation_20260821_%a.out
#SBATCH -e slurm_ExpAugmentation_20260821_%a.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Data_augmentation
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python Augmentation_expalone_circularcolonies.py