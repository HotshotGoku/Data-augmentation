"""Generate Flux augmenter synth for the colony-feature DOWNSTREAM experiment, using the winning
diversity config (cn0.65 + PG0.6). Sources = every canon train-pool rep NOT in the held-out test set;
K synth per source in PG-coupled batches -> {cond}_{tag}_s{n}.png (the downstream harness contract).
Env: CN_CKPT, GEN_OUT, REAL_DIR[canon], EXCLUDE_DIR[test/reals], K[8], BATCH[4], CN_SCALE[0.65],
     PG_ALPHA[0.6], CN_LAYERS[2], RES[256], STEPS[28], GUID[3.5]."""
import os, glob, cv2, torch
from PIL import Image
from diffusers import FluxPipeline, FluxControlNetPipeline, FluxControlNetModel
from optimum.quanto import quantize, freeze, qfloat8

BASE = os.environ.get("FLUX_BASE", "black-forest-labs/FLUX.1-dev")
CN_CKPT = os.environ["CN_CKPT"]; OUT = os.environ["GEN_OUT"]
REAL_DIR = os.environ.get("REAL_DIR", "/hpc/group/youlab/sa603/data/branching_256/canon")
EXCLUDE_DIR = os.environ.get("EXCLUDE_DIR", "/hpc/group/youlab/sa603/data/branching_256/test/reals")
K = int(os.environ.get("K", "8")); BATCH = int(os.environ.get("BATCH", "4"))
CN_SCALE = float(os.environ.get("CN_SCALE", "0.65")); PG_ALPHA = float(os.environ.get("PG_ALPHA", "0.6"))
CN_LAYERS = int(os.environ.get("CN_LAYERS", "2")); RES = int(os.environ.get("RES", "256"))
STEPS = int(os.environ.get("STEPS", "28")); GUID = float(os.environ.get("GUID", "3.5"))
os.makedirs(OUT, exist_ok=True); DT = torch.bfloat16

pipe0 = FluxPipeline.from_pretrained(BASE, torch_dtype=DT)
cn = FluxControlNetModel.from_transformer(pipe0.transformer, num_layers=CN_LAYERS, num_single_layers=0)
cn.load_state_dict(torch.load(CN_CKPT, map_location="cpu")); cn.to(DT)
pipe = FluxControlNetPipeline(vae=pipe0.vae, text_encoder=pipe0.text_encoder, text_encoder_2=pipe0.text_encoder_2,
    tokenizer=pipe0.tokenizer, tokenizer_2=pipe0.tokenizer_2, transformer=pipe0.transformer, controlnet=cn, scheduler=pipe0.scheduler)
quantize(pipe.transformer, weights=qfloat8); freeze(pipe.transformer); pipe.to("cuda")

def pg_callback(pipe, step, t, kw):   # PG repulsion (fixed scaling), same as flux_gen_diverse.py
    lat = kw["latents"]
    if PG_ALPHA > 0 and lat.shape[0] > 1:
        x = lat.flatten(1).float(); d = torch.cdist(x, x); h = d[d > 0].median() + 1e-6
        w = torch.exp(-(d ** 2) / (2 * h * h))
        rep = torch.stack([(w[i].unsqueeze(1) * (x[i:i+1] - x)).sum(0) for i in range(x.shape[0])])
        rep = rep / (rep.norm(dim=1, keepdim=True) + 1e-8)
        scale = x.norm(dim=1, keepdim=True) / (x.shape[1] ** 0.5)
        kw["latents"] = (lat + PG_ALPHA * (scale * rep).view_as(lat)).to(lat.dtype)
    return kw

def cond_of(p): return os.path.basename(p).split("_")[0]
exclude = {cond_of(p) for p in glob.glob(f"{EXCLUDE_DIR}/*.TIF")}
srcs = []
for fp in sorted(glob.glob(f"{REAL_DIR}/*_Rep*.TIF")):
    c = cond_of(fp)
    if c in exclude: continue
    tag = os.path.basename(fp).replace(".TIF", "").split("_", 1)[1]
    srcs.append((c, tag, fp))
print(f"[flux-dsynth] {len(srcs)} train-pool sources x {K} (cn_scale={CN_SCALE} pg={PG_ALPHA}) -> {OUT}", flush=True)

for k, (cond, tag, fp) in enumerate(srcs):
    im = cv2.resize(cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (RES, RES))
    ctrl = Image.fromarray(im); n = 0
    while n < K:
        b = min(BATCH, K - n)
        imgs = pipe(prompt="", control_image=ctrl, controlnet_conditioning_scale=CN_SCALE, num_inference_steps=STEPS,
                    guidance_scale=GUID, height=RES, width=RES, num_images_per_prompt=b,
                    generator=[torch.Generator("cpu").manual_seed(729 + n + i) for i in range(b)],
                    callback_on_step_end=pg_callback, callback_on_step_end_tensor_inputs=["latents"]).images
        for img in imgs:
            img.save(f"{OUT}/{cond}_{tag}_s{n}.png"); n += 1
    if k % 30 == 0: print(f"  {k}/{len(srcs)} {cond}_{tag}", flush=True)
print("[flux-dsynth] done", flush=True)
