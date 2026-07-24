# pipeline.py

from local_config import CKPT_PATH_V4
import shared_resources_config
from shared_resources_config import CONTROL_SD15_CKPT, CLDM_V15_YAML, CONFIG_RESOURCES

import os
import random 
import cv2
import numpy as np
import torch 
import einops   
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from cldm.cldm      import ControlLDM
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image, HWC3



######### ARG PARSE STUFF WHEN USING PARALLEL JOBS IGNORE FOR NOW #####
# import argparse
# # argument parsing, passing the folder name
# parser = argparse.ArgumentParser()
# parser.add_argument('--specific_folder', type=str, default='simtoexp', help='specific folder from the trained model saved locations')
# args = parser.parse_args()
# specific_folder=args.specific_folder


# def build(specific_folder: str):
#     ckpt_path = CKPT_PATH
#     config_yaml = OmegaConf.load(yaml_config)
#     params = OmegaConf.to_container(config_yaml.model.params, resolve=True)
#     model = ControlLDM.load_from_checkpoint(ckpt_path, **params).cuda()
#     ddim_sampler = DDIMSampler(model)
#     return model, ddim_sampler

##########################################################################


# determinism, seed, CUBLAS etc. all here, at module top
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark   = False
torch.use_deterministic_algorithms(True)

yaml_config = CLDM_V15_YAML          # YAML configuration file

# ------------------------------
# 1. Load the Model from Checkpoint
# ------------------------------


config_yaml = OmegaConf.load(yaml_config)
params = OmegaConf.to_container(config_yaml.model.params, resolve=True)
model = ControlLDM.load_from_checkpoint(CKPT_PATH_V4, **params).cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        print(type(img))
        # detected_map = np.zeros_like(img, dtype=np.uint8)
        # detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if CONFIG_RESOURCES.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if CONFIG_RESOURCES.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)

        if CONFIG_RESOURCES.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results
   

