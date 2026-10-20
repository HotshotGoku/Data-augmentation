
from datetime import datetime
import os
currentMinute = datetime.now().minute
currentHour   = datetime.now().hour
currentDay    = datetime.now().day
currentMonth  = datetime.now().month
currentYear   = datetime.now().year
# Path to the original Physics_constrained_DL_pattern_prediction project
# This can also be set via environment variable: PHYSICS_DL_PROJECT_PATH
ORIGINAL_PROJECT_BASE = '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction'

# Data directories for your augmentation experiments
BASE_FOLDER = '/hpc/group/youlab/ks723/storage'
SIM_IMAGES_FOLDER = '/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_100AUG'
EXP_IMAGES_FOLDER = '/hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed_100AUG'

# checkpoint path for inference 
CKPT_PATH= '/hpc/dctrl/ks723/Data_augmentation/lightning_logs/version_41457801/checkpoints/epoch=4-step=32124.ckpt'
# CKPT_PATH_V2 = '/hpc/dctrl/ks723/Data_augmentation/lightning_logs/version_42248443/checkpoints/epoch=2-step=81299.ckpt'
CKPT_PATH_V2= '/hpc/dctrl/ks723/Data_augmentation/lightning_logs/version_42248443/checkpoints/epoch=4-step=135499.ckpt'
CKPT_PATH_V3= '/hpc/dctrl/ks723/Data_augmentation/lightning_logs/version_42603802/checkpoints/epoch=4-step=124999.ckpt'
CKPT_PATH_V4 = os.environ.get("FT_EVAL_CKPT") or '/hpc/group/youlab/sa603/code/Data_augmentation/lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt' # sa603 full-data model (job 52034860). Set env FT_EVAL_CKPT to override for eval sweeps. pipeline.py loads THIS var.

# Named checkpoint registry for the sa603 models (repo-relative so it is not path-brittle).
# Import: `from Data_augmentation.utils.local_config import CHECKPOINTS` then CHECKPOINTS["generalist"].
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in utils/)
CHECKPOINTS = {
    "base":        f"{_REPO}/lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt",
    "2sp":         f"{_REPO}/finetune_runs/shallowsweep_lr5e-6/lightning_logs/version_53408286/checkpoints/epoch=3-step=75999.ckpt",
    "rattray":     f"{_REPO}/rattray_runs/rat_epochcurve/lightning_logs/version_54107493/checkpoints/epoch=3-step=719.ckpt",
    "multiplexed": f"{_REPO}/rattray_runs/multiplexed_ft/lightning_logs/version_54549115/checkpoints/epoch=2-step=1448.ckpt",
    "generalist":  f"{_REPO}/rattray_runs/generalist/lightning_logs/version_54592625/checkpoints/epoch=3-step=5723.ckpt",
}

# for saving 

OUTPUT_DIR_REPTOREP = f"/hpc/group/youlab/sa603/code/Data_augmentation/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_REPTOREP"
EXP_FOLDER_TEST= "/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3"

# for Nan's mutant library for the data collected by Dongheon and Kristen

MUTANT_EXP_FOLDER= '/hpc/group/youlab/ks723/storage/Exp_images/NL_evolution_library_Image'  
MUTANT_EXP_FOLDER_AUG= '/hpc/group/youlab/ks723/storage/Exp_images/NL_evolution_library_Image_AUG100'  # this has all images except the KuiZhu library strains
MUTANT_EXP_FOLDER_AUG_TRAINVALONLY= '/hpc/group/youlab/ks723/storage/Exp_images/NL_evolution_library_Image_AUG100_TrainValOnlyAFTER_RES_2000_FIX'   # note for this I moved some items out of the folder to incorportate in the test set # also fixed resolution issue
MUTANT_EXP_FOLDER_NOAUG= '/hpc/group/youlab/ks723/storage/Exp_images/NL_evolution_library_Image_preprocess_noaug'

EXP_FOLDER_KS= '/hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed'
EXP_FOLDER_KS_NOAUG = '/hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed_preprocess_noaug'

# create addtional test sets 

EXP_FOLDER_PRE_TEST_V2= '/hpc/group/youlab/ks723/storage/Exp_images/NL_evolution_library_Image/additional_strain_from_KuiZhu_PlosBiology'
EXP_FOLDER_PRE_TEST_V3= '/hpc/group/youlab/ks723/storage/Exp_images/NL_evolution_library_Image/additional_strain_from_KuiZhu_PlosBiology_plustestset'
EXP_FOLDER_TEST_V2= '/hpc/group/youlab/ks723/storage/Exp_images/NL_evolution_library_Image/additional_strain_from_KuiZhu_PlosBiology_preprocess_noaug'
EXP_FOLDER_TEST_V3= '/hpc/group/youlab/ks723/storage/Exp_images/NL_evolution_library_Image/additional_strain_from_KuiZhu_PlosBiology_plustestset_preprocess_noaug'  # has some additional images that were before in thre trainval set 

# Emrah's data 

EMRAH_EXP_FOLDER= '/hpc/group/youlab/ks723/storage/Exp_images/EmrahPaKp_dataset_renamed'
EMRAH_EXP_FOLDER_AUG = '/hpc/group/youlab/ks723/storage/Exp_images/EmrahPaKp_dataset_renamed_AUG100_TrainValOnly'
EMRAH_EXP_FOLDER_PRE_TEST=  '/hpc/group/youlab/ks723/storage/Exp_images/EmrahPaKp_dataset_renamed/Test_set'
EMRAH_EXP_FOLDER_TEST = '/hpc/group/youlab/ks723/storage/Exp_images/EmrahPaKp_dataset_renamed/Test_set_preprocess_noaug'

# Kristen's data 

KRISTEN_EXP_FOLDER_2SP= '/hpc/group/youlab/ks723/storage/Exp_images/KL_automatedcrops_2species'
KRISTEN_EXP_FOLDER_2SP_FILTERED= '/hpc/group/youlab/ks723/storage/Exp_images/KL_automatedcrops_2species_filtered' 
KRISTEN_EXP_FOLDER_2SP_FILTERED_RENAMED= '/hpc/group/youlab/ks723/storage/Exp_images/KL_automatedcrops_2species_filtered_renamed'
KRISTEN_EXP_FOLDER_2SP_FINALAUG= '/hpc/group/youlab/ks723/storage/Exp_images/KL_automatedcrops_2species_filtered_renamed_AUG100'

# differnt prompt json file 

JSON_FILE_3DATASETS_100000= os.path.join(BASE_FOLDER, f"prompt_experiments_3datasets_permutations_balanced_100000.json")
JSON_FILE_3DATASETS_FULL = os.path.join(BASE_FOLDER, f"prompt_experiments_3datasets_permutations_resfix.json")

OUTPUT_DIR_REPTOREP_SAVED= '/hpc/dctrl/ks723/Data_augmentation/inference/v202629_1921_REPTOREP'

TEST_FOLDER_MULTIPLEXED_SENSING= '/hpc/group/youlab/ks723/storage/Exp_images/Multiplexed_patterning/20260717_testset'
INFERENCE_FOLDER_MULTIPLEXED_SENSING= '/hpc/dctrl/ks723/Data_augmentation/inference/v2026724_1322_REPTOREP'

