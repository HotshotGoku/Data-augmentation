"""Flux pilot - diversity generation with CADS and/or Particle Guidance, reusing the validated
FluxControlNetPipeline so sampling stays correct.
  CADS: monkeypatch controlnet.forward to add timestep-annealed noise to controlnet_cond
        (gamma=1 clean late, 0 noised early) -> ŷ = sqrt(g)*c + noise*sqrt(1-g)*randn. arXiv 2310.17347.
  PG  : callback_on_step_end adds RBF batch-repulsion among the N per-source samples. arXiv 2310.13102.
Env: CN_CKPT, GEN_OUT, METHOD[none|cads|pg|both], CADS_NOISE[0.8], TAU1[0.6], TAU2[0.9],
     PG_ALPHA[0.0], CN_LAYERS[2], RES[256], NUM_SAMPLES[6], STEPS[28], GUID[3.5], CN_SCALE[1.0], MAX_SRC, OFFLOAD[none]."""
import os, glob, cv2, torch
from PIL import Image
from diffusers import FluxPipeline, FluxControlNetPipeline, FluxControlNetModel
from optimum.quanto import quantize, freeze, qfloat8

BASE = os.environ.get("FLUX_BASE", "black-forest-labs/FLUX.1-dev")
CN_CKPT = os.environ["CN_CKPT"]; OUT = os.environ["GEN_OUT"]
METHOD = os.environ.get("METHOD", "none")
CADS_NOISE = float(os.environ.get("CADS_NOISE", "0.8")); TAU1 = float(os.environ.get("TAU1", "0.6")); TAU2 = float(os.environ.get("TAU2", "0.9"))
PG_ALPHA = float(os.environ.get("PG_ALPHA", "0.0"))
SOURCES = os.environ.get("SOURCES", "/hpc/group/youlab/sa603/data/branching_256/test/sources")
CN_LAYERS = int(os.environ.get("CN_LAYERS", "2")); RES = int(os.environ.get("RES", "256"))
N = int(os.environ.get("NUM_SAMPLES", "6")); STEPS = int(os.environ.get("STEPS", "28"))
GUID = float(os.environ.get("GUID", "3.5")); CN_SCALE = float(os.environ.get("CN_SCALE", "1.0"))
os.makedirs(OUT, exist_ok=True); DT = torch.bfloat16

pipe0 = FluxPipeline.from_pretrained(BASE, torch_dtype=DT)
cn = FluxControlNetModel.from_transformer(pipe0.transformer, num_layers=CN_LAYERS, num_single_layers=0)
cn.load_state_dict(torch.load(CN_CKPT, map_location="cpu")); cn.to(DT)
pipe = FluxControlNetPipeline(vae=pipe0.vae, text_encoder=pipe0.text_encoder, text_encoder_2=pipe0.text_encoder_2,
    tokenizer=pipe0.tokenizer, tokenizer_2=pipe0.tokenizer_2, transformer=pipe0.transformer, controlnet=cn, scheduler=pipe0.scheduler)
quantize(pipe.transformer, weights=qfloat8); freeze(pipe.transformer)
pipe.to("cuda") if os.environ.get("OFFLOAD", "none") == "none" else pipe.enable_model_cpu_offload()
print(f"[flux-div] METHOD={METHOD} cads_noise={CADS_NOISE} pg_alpha={PG_ALPHA} cn_scale={CN_SCALE} guid={GUID}", flush=True)

# --- CADS: anneal noise into controlnet_cond based on the timestep the CN receives ---
def cads_gamma(t):
    t = t / 1000.0 if t > 1.0 else t          # Flux passes timestep/1000 in [0,1]
    if t <= TAU1: return 1.0
    if t >= TAU2: return 0.0
    return (TAU2 - t) / (TAU2 - TAU1)
if METHOD in ("cads", "both"):
    _orig = cn.forward
    def _patched(*a, **kw):
        c = kw.get("controlnet_cond"); ts = kw.get("timestep")
        if c is not None and ts is not None:
            g = cads_gamma(float(ts.flatten()[0].item()))
            kw["controlnet_cond"] = (g ** 0.5) * c + CADS_NOISE * ((1 - g) ** 0.5) * torch.randn_like(c)
        return _orig(*a, **kw)
    cn.forward = _patched

# --- PG: RBF repulsion among the N samples of one source, applied each step ---
def pg_callback(pipe, step, t, kw):
    lat = kw["latents"]
    if PG_ALPHA > 0 and lat.shape[0] > 1:
        x = lat.flatten(1).float()
        d = torch.cdist(x, x)
        h = d[d > 0].median() + 1e-6
        w = torch.exp(-(d ** 2) / (2 * h * h))
        rep = torch.stack([(w[i].unsqueeze(1) * (x[i:i+1] - x)).sum(0) for i in range(x.shape[0])])
        # FIX: don't divide by h^2 (vanishes in high-D). Normalize the repulsion DIRECTION and scale it
        # to a fraction (PG_ALPHA) of each sample's per-element magnitude, so alpha is interpretable.
        rep = rep / (rep.norm(dim=1, keepdim=True) + 1e-8)
        scale = x.norm(dim=1, keepdim=True) / (x.shape[1] ** 0.5)
        kw["latents"] = (lat + PG_ALPHA * (scale * rep).view_as(lat)).to(lat.dtype)
    return kw
cb = pg_callback if METHOD in ("pg", "both") and PG_ALPHA > 0 else None

srcs = sorted(glob.glob(f"{SOURCES}/*.TIF"))
MAX_SRC = int(os.environ.get("MAX_SRC", "0"))
if MAX_SRC: srcs = srcs[:MAX_SRC]
print(f"[flux-div] {len(srcs)} sources x {N} -> {OUT}", flush=True)
for k, fp in enumerate(srcs):
    prefix = os.path.basename(fp).split("_")[0]
    im = cv2.resize(cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (RES, RES))
    ctrl = Image.fromarray(im)
    imgs = pipe(prompt="", control_image=ctrl, controlnet_conditioning_scale=CN_SCALE, num_inference_steps=STEPS,
                guidance_scale=GUID, height=RES, width=RES, num_images_per_prompt=N,
                generator=[torch.Generator("cpu").manual_seed(i) for i in range(N)],
                callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"] if cb else None).images
    for i, img in enumerate(imgs):
        img.save(f"{OUT}/{prefix}_{i+1}.png")
    if k % 4 == 0: print(f"  {k}/{len(srcs)} {prefix}", flush=True)
print("[flux-div] done", flush=True)
