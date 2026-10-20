"""Generate multiplexed replicates from a fine-tuned PixCell Cell-ControlNet checkpoint, for scoring.
Loads the released Cell-ControlNet pipeline (fp32, bundled VAE, null UNI), overlays our fine-tuned
controlnet weights, and generates N seed-varied samples per frozen test source into {prefix}_{n}.png
(the same contract eval_metrics.py consumes). guidance_scale=1.0 => source-conditioned, seed-driven
diversity (higher guidance would amplify control and REDUCE diversity, which we don't want).
Run in the pixcell env; PYTHONPATH must include .../PixCell/controlnet.
Env: PIXCELL_CN_CKPT (required), GEN_OUT (required), SOURCES, NUM_SAMPLES[2], STEPS[50], GUID[1.0]."""
import os, sys, glob
sys.path.insert(0, "/hpc/group/youlab/sa603/code/PixCell/controlnet")
import cv2, torch
from diffusers import DiffusionPipeline

CKPT = os.environ["PIXCELL_CN_CKPT"]
OUT = os.environ["GEN_OUT"]
SOURCES = os.environ.get("SOURCES", "/hpc/group/youlab/sa603/data/multiplexed_eval/sources")
N = int(os.environ.get("NUM_SAMPLES", "2"))
STEPS = int(os.environ.get("STEPS", "50"))
GUID = float(os.environ.get("GUID", "1.0"))
os.makedirs(OUT, exist_ok=True)

pipe = DiffusionPipeline.from_pretrained(
    "StonyBrook-CVLab/PixCell-256-Cell-ControlNet",
    custom_pipeline="StonyBrook-CVLab/PixCell-pipeline-ControlNet",
    trust_remote_code=True, torch_dtype=torch.float32).to("cuda")
sd = torch.load(CKPT, map_location="cpu")
missing, unexpected = pipe.controlnet.load_state_dict(sd, strict=False)
print(f"[gen] loaded {CKPT}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
pipe.controlnet.eval()

# Optional stochastic sampler swap (diversity boost). Default dpm = deterministic DPM-Solver++.
SAMPLER = os.environ.get("SAMPLER", "dpm")
ETA = float(os.environ.get("ETA", "0.0"))
if SAMPLER == "sde":
    from diffusers import DPMSolverMultistepScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
elif SAMPLER == "ddim":
    from diffusers import DDIMScheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
elif SAMPLER == "ddpm":
    from diffusers import DDPMScheduler
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
print(f"[gen] SAMPLER={SAMPLER} ETA={ETA} scheduler={type(pipe.scheduler).__name__}", flush=True)

uncond = pipe.get_unconditional_embedding(1)

srcs = {}
for fp in sorted(glob.glob(f"{SOURCES}/*.TIF")):
    srcs.setdefault(os.path.basename(fp).split("_")[0], fp)
print(f"[gen] {len(srcs)} sources -> {OUT} (N={N}, steps={STEPS}, guid={GUID})", flush=True)

for k, (prefix, fp) in enumerate(srcs.items()):
    im = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (256, 256))
    for i in range(1, N + 1):
        g = torch.Generator("cuda").manual_seed(729397049 + i)
        out = pipe(uni_embeds=uncond, negative_uni_embeds=uncond, controlnet_input=im,
                   guidance_scale=GUID, num_inference_steps=STEPS, num_images_per_prompt=1,
                   eta=ETA, generator=g).images[0]
        out.save(f"{OUT}/{prefix}_{i}.png")
    if k % 20 == 0:
        print(f"  {k}/{len(srcs)}", flush=True)
print("[gen] done", flush=True)
