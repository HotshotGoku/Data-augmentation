"""Flux ControlNet feasibility smoke test (Track B) — plumbing + VRAM check on a 24GB A5000.
Not a quality result (zero-shot Flux doesn't know colonies): confirms the pipeline runs, in what
offload mode, and peak VRAM — to estimate effort/feasibility of fine-tuning a Flux ControlNet.

Env: FLUX_BASE, FLUX_CN, SRC (control source img), OUT, HW, STEPS, CN_SCALE, GUIDANCE, OFFLOAD.
Requires the flowmatch env + HF_TOKEN with the FLUX.1-dev license accepted."""
import os, time, cv2, numpy as np, torch
from PIL import Image
from diffusers import FluxControlNetModel, FluxControlNetPipeline

BASE = os.environ.get("FLUX_BASE", "black-forest-labs/FLUX.1-dev")
CN = os.environ.get("FLUX_CN", "InstantX/FLUX.1-dev-Controlnet-Canny")
SRC = os.environ.get("SRC", "/hpc/group/youlab/sa603/data/multiplexed_eval/sources/R10C1_src.TIF")
OUT = os.environ.get("OUT", "/hpc/group/youlab/sa603/code/Data_augmentation/flux_smoke_out")
HW = int(os.environ.get("HW", "512"))
STEPS = int(os.environ.get("STEPS", "24"))
CN_SCALE = float(os.environ.get("CN_SCALE", "0.6"))
GUID = float(os.environ.get("GUIDANCE", "3.5"))
OFFLOAD = os.environ.get("OFFLOAD", "sequential")  # sequential (lowest VRAM) | model | none
os.makedirs(OUT, exist_ok=True)

# control image = canny edges of the source (structural control), upsized to HW
im = cv2.resize(cv2.cvtColor(cv2.imread(SRC, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (HW, HW))
canny = cv2.Canny(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), 80, 160)
ctrl = Image.fromarray(np.stack([canny] * 3, -1))
ctrl.save(f"{OUT}/control_canny.png"); Image.fromarray(im).save(f"{OUT}/source.png")

print(f"[flux] base={BASE}\n[flux] cn={CN}\n[flux] HW={HW} steps={STEPS} offload={OFFLOAD}", flush=True)
t0 = time.time()
controlnet = FluxControlNetModel.from_pretrained(CN, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(BASE, controlnet=controlnet, torch_dtype=torch.bfloat16)
if OFFLOAD == "sequential":
    pipe.enable_sequential_cpu_offload()
elif OFFLOAD == "model":
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")
print(f"[flux] loaded in {time.time()-t0:.0f}s", flush=True)

torch.cuda.reset_peak_memory_stats()
t1 = time.time()
img = pipe(prompt="", control_image=ctrl, controlnet_conditioning_scale=CN_SCALE,
           num_inference_steps=STEPS, guidance_scale=GUID, height=HW, width=HW,
           generator=torch.Generator("cpu").manual_seed(0)).images[0]
dt = time.time() - t1
img.save(f"{OUT}/flux_gen.png")
print(f"[flux] OK: gen in {dt:.0f}s | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB | saved {OUT}/flux_gen.png", flush=True)
