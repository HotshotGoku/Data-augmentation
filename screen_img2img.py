"""Generic backbone screen: AutoPipelineForImage2Image (one harness, many backbones) runs an SDEdit
strength sweep on the 28 branching test sources. Prior-fit + img2img-diversity screen (zero-shot, no
training) -> fast ranking of which backbones are worth the full ControlNet-training treatment.
Env: NAME, MODEL_ID, DTYPE[fp16|bf16], STRENGTHS[0.5,0.7], NUM[4], STEPS[30], GUID[5], OUTROOT, SOURCES, RES[256]."""
import os, glob, traceback, torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image

NAME = os.environ["NAME"]; MID = os.environ["MODEL_ID"]
DT = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(os.environ.get("DTYPE", "fp16"), torch.float16)
STR = [float(x) for x in os.environ.get("STRENGTHS", "0.5,0.7").split(",")]
NUM = int(os.environ.get("NUM", "4")); STEPS = int(os.environ.get("STEPS", "30"))
GUID = float(os.environ.get("GUID", "5.0")); RES = int(os.environ.get("RES", "256"))
SRC = os.environ.get("SOURCES", "/hpc/group/youlab/sa603/data/branching_256/test/sources")
OUTROOT = os.environ.get("OUTROOT", "/hpc/group/youlab/sa603/code/Data_augmentation/backbone_screen")

print(f"[screen] {NAME} <- {MID} ({DT})", flush=True)
pipe = AutoPipelineForImage2Image.from_pretrained(MID, torch_dtype=DT)
try: pipe.set_progress_bar_config(disable=True)
except Exception: pass
try:
    pipe.to("cuda")                                  # fast path if it fits 24GB
    print("[screen] on GPU", flush=True)
except Exception:
    pipe.enable_model_cpu_offload(); print("[screen] cpu-offload fallback", flush=True)

srcs = sorted(glob.glob(f"{SRC}/*.TIF"))
for s in STR:
    out = f"{OUTROOT}/{NAME}_s{int(s*100)}"; os.makedirs(out, exist_ok=True)
    for fp in srcs:
        cond = os.path.basename(fp).split("_")[0]
        img = Image.open(fp).convert("RGB").resize((RES, RES))
        for i in range(NUM):
            g = torch.Generator("cpu").manual_seed(i)
            try:
                r = pipe(prompt="", image=img, strength=s, num_inference_steps=STEPS,
                         guidance_scale=GUID, height=RES, width=RES, generator=g)
            except TypeError:                         # some pipelines reject height/width on img2img
                r = pipe(prompt="", image=img, strength=s, num_inference_steps=STEPS,
                         guidance_scale=GUID, generator=g)
            r.images[0].resize((RES, RES)).save(f"{out}/{cond}_{i+1}.png")
    print(f"[screen] {NAME} strength {s} -> {out} ({len(srcs)*NUM} imgs)", flush=True)
print("SCREENDONE", NAME, flush=True)
