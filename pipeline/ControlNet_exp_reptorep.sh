#!/bin/bash 
#SBATCH -o slurm_ControlNet_exp_reptorep_20260202_Full_%a.out
#SBATCH -e slurm_ControlNet_exp_reptorep_20260202_Full_%a.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Data_augmentation
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python reptorep_train.py