"""Generate multiplexed replicates from a fine-tuned PixCell-1024 LoRA ControlNet, for scoring.
Rebuilds the from-base ControlNet + ControlNet-aware denoiser EXACTLY as training did (config-based,
warm-init from base), wraps the ControlNet in the same LoRA config, loads the fine-tuned weights, and
runs a manual eps-prediction sampling loop at 1024px (source upscaled to 1024). Each sample is then
downscaled to 256 and written as {prefix}_{n}.png (the contract eval_metrics.py consumes). Null UNI,
control via the ControlNet branch; diversity comes from seed-varied initial noise. Manual loop is used
because PixCell-1024 has no released ControlNet pipeline (unlike the 256 repo). Run in pixcell env;
PYTHONPATH includes .../PixCell/controlnet.
Handles PixCell-1024 and PixCell-256 via BASE_MODEL + RES (must match the checkpoint's training).
Env: PIXCELL_CN_CKPT(req) GEN_OUT(req) SOURCES BASE_MODEL[PixCell-1024] RES[1024] NUM_SAMPLES[2] STEPS[50] SAMPLER[ddim] ETA[0.0]."""
import os, sys, glob, inspect
sys.path.insert(0, "/hpc/group/youlab/sa603/code/PixCell/controlnet")
import cv2, torch
from diffusers import DiffusionPipeline, AutoencoderKL
from pixcell_controlnet import PixCellControlNet
from pixcell_controlnet_transformer import PixCellTransformer2DModelControlNet

CKPT = os.environ["PIXCELL_CN_CKPT"]
OUT = os.environ["GEN_OUT"]
SOURCES = os.environ.get("SOURCES", "/hpc/group/youlab/sa603/data/multiplexed_eval/sources")
N = int(os.environ.get("NUM_SAMPLES", "2"))
STEPS = int(os.environ.get("STEPS", "50"))
SAMPLER = os.environ.get("SAMPLER", "ddim")
ETA = float(os.environ.get("ETA", "0.0"))
BASE_MODEL = os.environ.get("BASE_MODEL", "StonyBrook-CVLab/PixCell-1024")  # set PixCell-256 for the same-recipe 256 control
RES = int(os.environ.get("RES", "1024"))
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

vae = AutoencoderKL.from_pretrained("StonyBrook-CVLab/PixCell-256-Cell-ControlNet", subfolder="vae")
pipe = DiffusionPipeline.from_pretrained(BASE_MODEL, vae=vae,
    custom_pipeline="StonyBrook-CVLab/PixCell-pipeline", trust_remote_code=True,
    low_cpu_mem_usage=False, device_map=None)
sched0 = pipe.scheduler
vae_scale = pipe.vae.config.scaling_factor
vae_shift = getattr(pipe.vae.config, "shift_factor", 0)

# Rebuild ControlNet + ControlNet-aware denoiser EXACTLY as training (config-based, warm-init from base)
bcfg = dict(pipe.transformer.config)
base_sd = pipe.transformer.state_dict()
def cfg_for(cls):
    ok = set(inspect.signature(cls.__init__).parameters) - {"self", "args", "kwargs"}
    return {k: v for k, v in bcfg.items() if k in ok}
cn_kwargs = cfg_for(PixCellControlNet); cn_kwargs["n_controlnet_blocks"] = 27
controlnet = PixCellControlNet(**cn_kwargs)
controlnet.load_state_dict(base_sd, strict=False)
cn_transformer = PixCellTransformer2DModelControlNet(**cfg_for(PixCellTransformer2DModelControlNet))
cn_transformer.load_state_dict(base_sd, strict=False)
pipe.transformer = cn_transformer

# same LoRA wrap as training, then overlay the fine-tuned weights
from peft import LoraConfig, get_peft_model
controlnet = get_peft_model(controlnet, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.0))
sd = torch.load(CKPT, map_location="cpu")
missing, unexpected = controlnet.load_state_dict(sd, strict=False)
print(f"[gen1024] loaded {CKPT}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

vae = pipe.vae.to(dev).eval()
transformer = pipe.transformer.to(dev).eval()
controlnet = controlnet.to(dev).eval()

if SAMPLER == "ddim":
    from diffusers import DDIMScheduler; scheduler = DDIMScheduler.from_config(sched0.config)
elif SAMPLER == "ddpm":
    from diffusers import DDPMScheduler; scheduler = DDPMScheduler.from_config(sched0.config)
elif SAMPLER == "dpm":
    from diffusers import DPMSolverMultistepScheduler; scheduler = DPMSolverMultistepScheduler.from_config(sched0.config)
elif SAMPLER == "sde":
    from diffusers import DPMSolverMultistepScheduler; scheduler = DPMSolverMultistepScheduler.from_config(sched0.config, algorithm_type="sde-dpmsolver++")
else:
    scheduler = sched0
scheduler.set_timesteps(STEPS, device=dev)
print(f"[gen1024] SAMPLER={SAMPLER} steps={STEPS} scheduler={type(scheduler).__name__}", flush=True)

try:
    null_uni = pipe.get_unconditional_embedding(1).to(dev)
except Exception:
    null_uni = pipe.transformer.caption_projection.uncond_embedding.clone().reshape(1, 1, -1).to(dev)

def encode(img_bchw):
    return (vae.encode((2 * img_bchw - 1).to(dev)).latent_dist.mean - vae_shift) * vae_scale

C = vae.config.latent_channels
srcs = {}
for fp in sorted(glob.glob(f"{SOURCES}/*.TIF")):
    srcs.setdefault(os.path.basename(fp).split("_")[0], fp)
print(f"[gen1024] {len(srcs)} sources -> {OUT} (N={N}, C={C}, res={RES})", flush=True)

for k, (prefix, fp) in enumerate(srcs.items()):
    im = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (RES, RES))
    src = torch.from_numpy(im).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        sl = encode(src)
    uni = null_uni.expand(1, -1, -1)
    for i in range(1, N + 1):
        g = torch.Generator(dev).manual_seed(729397049 + i)
        x = torch.randn(1, C, RES // 8, RES // 8, generator=g, device=dev) * getattr(scheduler, "init_noise_sigma", 1.0)
        with torch.no_grad():
            for t in scheduler.timesteps:
                tt = t.reshape(1).to(dev)
                co = controlnet(hidden_states=x, conditioning=sl, encoder_hidden_states=uni, timestep=tt, return_dict=False)[0]
                eps = transformer(x, encoder_hidden_states=uni, controlnet_outputs=co, timestep=tt, added_cond_kwargs={}, return_dict=False)[0][:, :16]
                kw = {"eta": ETA} if SAMPLER == "ddim" else {}
                try:
                    x = scheduler.step(eps, t, x, generator=g, **kw).prev_sample
                except TypeError:
                    x = scheduler.step(eps, t, x, **kw).prev_sample
            img = vae.decode(x / vae_scale + vae_shift).sample
        img = ((img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(f"{OUT}/{prefix}_{i}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if k % 20 == 0:
        print(f"  {k}/{len(srcs)}", flush=True)
print("[gen1024] done", flush=True)
