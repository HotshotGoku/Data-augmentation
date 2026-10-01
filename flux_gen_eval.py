"""Flux pilot - generate replicates from a trained Flux ControlNet, for scoring on the frozen bench.
Loads base (fp8) + the trained CN, generates NUM_SAMPLES per test source -> {prefix}_{i}.png (the
eval_metrics.py contract). One transformer is shared between the CN build and the pipeline.
Env: CN_CKPT(req), GEN_OUT(req), SOURCES, CN_LAYERS[2], RES[256], NUM_SAMPLES[4], STEPS[28], GUID[3.5], CN_SCALE[1.0]."""
import os, glob, cv2, torch
from PIL import Image
from diffusers import FluxPipeline, FluxControlNetPipeline, FluxControlNetModel
from optimum.quanto import quantize, freeze, qfloat8

BASE = os.environ.get("FLUX_BASE", "black-forest-labs/FLUX.1-dev")
CN_CKPT = os.environ["CN_CKPT"]; OUT = os.environ["GEN_OUT"]
SOURCES = os.environ.get("SOURCES", "/hpc/group/youlab/sa603/data/branching_256/test/sources")
CN_LAYERS = int(os.environ.get("CN_LAYERS", "2")); RES = int(os.environ.get("RES", "256"))
N = int(os.environ.get("NUM_SAMPLES", "4")); STEPS = int(os.environ.get("STEPS", "28"))
GUID = float(os.environ.get("GUID", "3.5")); CN_SCALE = float(os.environ.get("CN_SCALE", "1.0"))
os.makedirs(OUT, exist_ok=True)
DT = torch.bfloat16

pipe0 = FluxPipeline.from_pretrained(BASE, torch_dtype=DT)   # source of vae/text/transformer
cn = FluxControlNetModel.from_transformer(pipe0.transformer, num_layers=CN_LAYERS, num_single_layers=0)
cn.load_state_dict(torch.load(CN_CKPT, map_location="cpu")); cn.to(DT)
print(f"[flux-gen] loaded CN {CN_CKPT}", flush=True)
pipe = FluxControlNetPipeline(
    vae=pipe0.vae, text_encoder=pipe0.text_encoder, text_encoder_2=pipe0.text_encoder_2,
    tokenizer=pipe0.tokenizer, tokenizer_2=pipe0.tokenizer_2,
    transformer=pipe0.transformer, controlnet=cn, scheduler=pipe0.scheduler)
quantize(pipe.transformer, weights=qfloat8); freeze(pipe.transformer)
if os.environ.get("OFFLOAD", "model") == "none":
    pipe.to("cuda")            # faster at 256px if it fits (fp8 base ~12.5G + small activations)
else:
    pipe.enable_model_cpu_offload()

srcs = sorted(glob.glob(f"{SOURCES}/*.TIF"))
MAX_SRC = int(os.environ.get("MAX_SRC", "0"))
if MAX_SRC: srcs = srcs[:MAX_SRC]
print(f"[flux-gen] {len(srcs)} sources x {N} -> {OUT}", flush=True)
for k, fp in enumerate(srcs):
    prefix = os.path.basename(fp).split("_")[0]
    im = cv2.resize(cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (RES, RES))
    ctrl = Image.fromarray(im)
    for i in range(N):
        img = pipe(prompt="", control_image=ctrl, controlnet_conditioning_scale=CN_SCALE,
                   num_inference_steps=STEPS, guidance_scale=GUID, height=RES, width=RES,
                   generator=torch.Generator("cpu").manual_seed(i)).images[0]
        img.save(f"{OUT}/{prefix}_{i+1}.png")
    if k % 5 == 0:
        print(f"  {k}/{len(srcs)} {prefix}", flush=True)
print("[flux-gen] done", flush=True)
