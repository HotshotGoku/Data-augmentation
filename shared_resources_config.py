"""
Shared Resources - points to original project files without duplication.
Set PHYSICS_DL_PROJECT_PATH env var or edit ORIGINAL_PROJECT_BASE below.

"""

import sys
import os

# Update this path for your setup or use env var
ORIGINAL_PROJECT_BASE = os.environ.get(
    'PHYSICS_DL_PROJECT_PATH',
    '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction'
)

# Add original project to Python path
CONTROLNET_PATH = f"{ORIGINAL_PROJECT_BASE}/sim_to_exp_diffusion/controlnet_essential"
if CONTROLNET_PATH not in sys.path:
    sys.path.insert(0, CONTROLNET_PATH)

import config as CONFIG_RESOURCES

# Checkpoint files
CONTROL_SD15_CKPT = f"{CONTROLNET_PATH}/control_sd15_ini.ckpt"
V1_5_PRUNED_CKPT = f"{CONTROLNET_PATH}/models/v1-5-pruned.ckpt"
CLDM_V15_YAML = f"{CONTROLNET_PATH}/models/cldm_v15.yaml"