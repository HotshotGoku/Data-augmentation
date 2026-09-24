"""De-risk gate for PixCell-256: does the model + its Cell-ControlNet LOAD and RUN with our
source-image conditioning + the learned null UNI token? Zero-shot (no fine-tune), so output is
EXPECTED to look pathology-ish / not colony-like -- the point is to confirm (a) the env/deps load
the pipeline, (b) the real __call__ API, (c) the source->control wiring runs and yields images.
Run in the `pixcell` env (diffusers 0.32.2 + timm)."""
import os, sys, glob, inspect, traceback
import numpy as np, cv2, torch
from PIL import Image
from diffusers import DiffusionPipeline

OUT = os.environ.get("OUT", "/hpc/group/youlab/sa603/code/Data_augmentation/backbone_derisk_out")
os.makedirs(OUT, exist_ok=True)
SRC = sorted(glob.glob("/hpc/group/youlab/sa603/data/multiplexed_eval/sources/*.TIF"))[0]
print(f"[pixcell] source = {SRC}", flush=True)

print("[pixcell] loading pipeline (trust_remote_code, fp16) ...", flush=True)
pipe = DiffusionPipeline.from_pretrained(
    "StonyBrook-CVLab/PixCell-256-Cell-ControlNet",
    custom_pipeline="StonyBrook-CVLab/PixCell-pipeline-ControlNet",
    trust_remote_code=True, torch_dtype=torch.float16).to("cuda")
print("[pixcell] LOADED.", flush=True)
try:
    print("[pixcell] __call__ signature:", inspect.signature(pipe.__call__), flush=True)
except Exception as e:
    print("[pixcell] signature introspection failed:", e, flush=True)
print("[pixcell] has get_unconditional_embedding:", hasattr(pipe, "get_unconditional_embedding"), flush=True)

im = cv2.cvtColor(cv2.imread(SRC, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (256, 256))
pil = Image.fromarray(im)

uncond = None
try:
    uncond = pipe.get_unconditional_embedding(1)
    print("[pixcell] null uni_embeds shape/dtype:", tuple(uncond.shape), uncond.dtype, flush=True)
except Exception as e:
    print("[pixcell] get_unconditional_embedding FAILED:", repr(e), flush=True)

def gen(ctrl, tag):
    saved = []
    for s in range(2):
        g = torch.Generator("cuda").manual_seed(1000 + s)
        out = pipe(uni_embeds=uncond, negative_uni_embeds=uncond, controlnet_input=ctrl,
                   guidance_scale=1.5, num_inference_steps=20, generator=g).images[0]
        p = f"{OUT}/pixcell_smoke_{tag}_{s}.png"
        out.save(p); saved.append(p)
    return saved

ok = False
for tag, ctrl in [("pil", pil), ("np", im)]:
    try:
        saved = gen(ctrl, tag)
        print(f"[pixcell] GEN OK with controlnet_input={tag} -> {saved}", flush=True)
        ok = True
        break
    except Exception as e:
        print(f"[pixcell] gen FAILED with controlnet_input={tag}: {repr(e)}", flush=True)
        traceback.print_exc()
print(f"[pixcell] smoke done (generation_ok={ok})", flush=True)
