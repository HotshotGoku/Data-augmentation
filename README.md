# Data Augmentation Project

This is a spinoff project from the main Physics_constrained_DL_pattern_prediction project.

## Shared Resources

This project uses shared resources (checkpoint files, datasets) from the original project without duplicating them.

The configuration for shared resources is in `shared_resources_config.py`.

### Key Shared Resources:

- **control_sd15_ini.ckpt**: Located in original project's sim_to_exp_diffusion/controlnet_essential/
- **v1-5-pruned.ckpt**: Located in original project's sim_to_exp_diffusion/controlnet_essential/models/
- **Trained models**: /hpc/group/youlab/ks723/miniconda3/saved_models/trained/
- **Datasets**: /hpc/group/youlab/ks723/storage/



### Usage

**Option 1: Import modules from original project (Recommended)**

The config automatically adds the original project to your Python path. You can import modules directly:

```python
# This is already done in shared_resources_config
import sys
sys.path.insert(0, '/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential')

# Now you can import from the original project
from cldm.model import create_model, load_state_dict
from shared_resources_config import CONTROL_SD15_CKPT, CLDM_V15_YAML

# Use the shared checkpoint and config
model = create_model(CLDM_V15_YAML).cpu()
model.load_state_dict(load_state_dict(CONTROL_SD15_CKPT, location='cpu'))
```

**Option 2: Copy only what you need**

If you prefer independence, copy the `cldm/` directory and other modules you need to this folder.

### Example Files

To check that all shared resources are accessible:

```bash
python shared_resources_config.py
```

## Getting Started

1. Use the existing conda environment:

   ```bash
   conda activate pytorch_PA_patternprediction
   ```
2. Set up your local configuration:

   ```bash
   cd /hpc/dctrl/ks723/Data_augmentation

   # Option 1: Use environment variable (recommended for GitHub)
   export PHYSICS_DL_PROJECT_PATH=/your/path/to/Physics_constrained_DL_pattern_prediction

   # Option 2: Edit shared_resources_config.py directly
   # (but this will be committed to git)
   ```
3. Verify shared resources:

   ```bash
   python shared_resources_config.py
   ```
4. Start developing your augmentation scripts!

### For GitHub

The project is GitHub-ready:

- `.gitignore` excludes checkpoints and local configs
- `example_config.py` shows configuration structure
- `shared_resources_config.py` uses environment variable `PHYSICS_DL_PROJECT_PATH`
- No hardcoded absolute paths in your scripts
