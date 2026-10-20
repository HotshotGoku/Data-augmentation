"""PixCell-256 zero-shot smoke v2 (fixed loader).

Fixes the earlier ImportError by putting the GitHub clone's controlnet/ dir on sys.path so the
custom pipeline's helper imports (pixcell_controlnet, pixcell_controlnet_transformer) resolve.
Uses the BUNDLED VAE (no SD3.5 gating) and the learned NULL uni token (no UNI encoder / no gating).
Feeds our SOURCE colony image into the ControlNet branch. Zero-shot (no fine-tune) => output is
EXPECTED to look pathology-ish; the point is to confirm the pipeline loads + runs our wiring."""
import os, sys, glob, inspect, traceback
sys.path.insert(0, "/hpc/group/youlab/sa603/code/PixCell/controlnet")  # pixcell_controlnet[/ _transformer / _transformer_2d]
import numpy as np, cv2, torch
from PIL import Image
from diffusers import DiffusionPipeline

OUT = os.environ.get("OUT", "/hpc/group/youlab/sa603/code/Data_augmentation/pixcell_smoke_out")
os.makedirs(OUT, exist_ok=True)
SRC = sorted(glob.glob("/hpc/group/youlab/sa603/data/multiplexed_eval/sources/*.TIF"))[0]
print(f"[pixcell2] source={SRC}", flush=True)

print("[pixcell2] loading pipeline (bundled VAE, trust_remote_code, fp16) ...", flush=True)
pipe = DiffusionPipeline.from_pretrained(
    "StonyBrook-CVLab/PixCell-256-Cell-ControlNet",
    custom_pipeline="StonyBrook-CVLab/PixCell-pipeline-ControlNet",
    trust_remote_code=True, torch_dtype=torch.float32).to("cuda")  # fp32: pipeline hardcodes .float() on controlnet_input
print("[pixcell2] LOADED. __call__ signature:", inspect.signature(pipe.__call__), flush=True)

im = cv2.cvtColor(cv2.imread(SRC, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (256, 256))

uncond = pipe.get_unconditional_embedding(1)
print("[pixcell2] null uni_embeds:", tuple(uncond.shape), uncond.dtype, flush=True)

def gen(ctrl, tag):
    saved = []
    for s in range(3):
        g = torch.Generator("cuda").manual_seed(1000 + s)
        out = pipe(uni_embeds=uncond, negative_uni_embeds=uncond, controlnet_input=ctrl,
                   guidance_scale=1.0, num_inference_steps=20, num_images_per_prompt=1,
                   generator=g).images[0]
        p = f"{OUT}/pixcell_smoke2_{tag}_{s}.png"; out.save(p); saved.append(p)
    return saved

ok = False
for tag, ctrl in [("np", im)]:   # pipeline expects a numpy uint8 HWC array (it does .copy()/255)
    try:
        print(f"[pixcell2] GEN OK controlnet_input={tag} -> {gen(ctrl, tag)}", flush=True)
        ok = True
        break
    except Exception as e:
        print(f"[pixcell2] gen FAILED controlnet_input={tag}: {repr(e)}", flush=True)
        traceback.print_exc()
print(f"[pixcell2] smoke done (generation_ok={ok})", flush=True)
