#!/bin/bash 
#SBATCH -o slurm_testscript_%j.out
#SBATCH -e slurm_testscript_%j.err
#SBATCH -p youlab-gpu
#SBATCH -w dcc-youlab-gpu-28
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Data_augmentation
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python test_script.py