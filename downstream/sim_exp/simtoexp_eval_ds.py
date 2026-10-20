"""Paired eval of a sim->exp checkpoint (background-black space): predict an experimental image
from each held-out test simulation, compare to the PAIRED real experimental image via SSIM (higher
better) + LPIPS (lower better). Same seed for every model -> fair A-vs-B comparison.
Env: DS_EVAL_CKPT (required), DS_EVAL_TAG. Run on a GPU node. Upload to Data_augmentation root."""
import os
import glob
import json
import csv
import cv2
import numpy as np
import torch
import einops
from pytorch_lightning import seed_everything
# NOTE: import shared_resources_config BEFORE any cldm import — it puts the Simulation project's
# controlnet_essential (which provides `cldm`) on sys.path. Otherwise: ModuleNotFoundError: cldm.
from Data_augmentation.utils.shared_resources_config import CLDM_V15_YAML
from Data_augmentation.utils import tracking
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.preprocess import preprocess_simulation_graybackground, preprocess_experimental_backgroundblack
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn

TAG = os.environ.get("DS_EVAL_TAG", "ds_eval")
CKPT = os.environ.get("DS_EVAL_CKPT")
if not CKPT:  # resolve latest checkpoint for this tag (enables dependency-chained eval after training)
    _c = sorted(glob.glob(f"/hpc/group/youlab/sa603/code/Data_augmentation/downstream_runs/{TAG}/lightning_logs/*/checkpoints/*.ckpt"))
    if not _c:
        raise SystemExit(f"[eval] no checkpoint found for tag {TAG}")
    CKPT = _c[-1]
    print(f"[eval] resolved ckpt for {TAG}: {CKPT}", flush=True)
OUT = os.environ.get("DS_EVAL_OUT", f"/hpc/group/youlab/sa603/code/Data_augmentation/downstream_eval/{TAG}")
SIM_TEST = "/hpc/group/youlab/sa603/data/sim_project/extracted/sim_to_exp_diffusion/SimcorrtoExp_testset"
EXP_TEST = "/hpc/group/youlab/sa603/data/sim_project/extracted/sim_to_exp_diffusion/Exp_testset"
DDIM_STEPS, SCALE, ETA, SEED = 30, 9.0, 0.0, 42
N_PROMPT = "longbody, lowres, bad anatomy, cropped, worst quality, low quality"
os.makedirs(os.path.join(OUT, "pred"), exist_ok=True)

model = create_model(CLDM_V15_YAML).cpu()
model.load_state_dict(load_state_dict(CKPT, location="cpu"))
model = model.cuda().eval()
sampler = DDIMSampler(model)
lp = lpips_lib.LPIPS(net="alex").cuda()


@torch.no_grad()
def predict(sim_rgb):  # sim_rgb: uint8 256x256x3 (graybackground)
    control = torch.from_numpy(sim_rgb.astype(np.float32) / 255.0).cuda()
    control = einops.rearrange(torch.stack([control], 0), "b h w c -> b c h w").clone()
    seed_everything(SEED)
    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([""])]}
    un = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([N_PROMPT])]}
    model.control_scales = [1.0] * 13
    samples, _ = sampler.sample(DDIM_STEPS, 1, (4, 32, 32), cond, verbose=False, eta=ETA,
                                unconditional_guidance_scale=SCALE, unconditional_conditioning=un)
    x = model.decode_first_stage(samples)
    x = (einops.rearrange(x, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return x[0]


rows, ssims, lps = [], [], []
for sp in sorted(glob.glob(os.path.join(SIM_TEST, "*.TIF"))):
    name = os.path.basename(sp)
    real_p = os.path.join(EXP_TEST, name)
    if not os.path.exists(real_p):
        continue
    sim = preprocess_simulation_graybackground(sp)
    sim = np.repeat(sim[:, :, None], 3, 2)
    pred = predict(sim)
    real = preprocess_experimental_backgroundblack(real_p)
    cv2.imwrite(os.path.join(OUT, "pred", name.replace(".TIF", ".png")), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
    s = float(ssim_fn(real, pred, channel_axis=2))
    a = torch.from_numpy(pred.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].cuda()
    b = torch.from_numpy(real.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].cuda()
    l = float(lp(a, b).item())
    rows.append((name, s, l)); ssims.append(s); lps.append(l)

res = {"tag": TAG, "ckpt": CKPT, "n": len(rows),
       "SSIM_mean": float(np.mean(ssims)) if ssims else None,
       "LPIPS_mean": float(np.mean(lps)) if lps else None}
json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
with open(os.path.join(OUT, "per_image.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["name", "SSIM", "LPIPS"]); w.writerows(sorted(rows))
tracking.write_manifest("simexp_eval", res)
print(f"[eval] {TAG}: SSIM {res['SSIM_mean']:.4f}  LPIPS {res['LPIPS_mean']:.4f}  (n={res['n']})", flush=True)
