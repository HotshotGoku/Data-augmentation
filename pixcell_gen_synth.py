"""Generate downstream-augmentation synth from a fine-tuned PixCell Cell-ControlNet, in the naming
the downstream decoder consumes: {cond}_src{rep}_s{n}.png from reps 8,9,10 (present in all 70 conds).
Mirrors gen_multiplexed_synth.py but uses the PixCell pipeline + our controlnet weights.
Run in pixcell env; PYTHONPATH includes .../PixCell/controlnet.
Env: PIXCELL_CN_CKPT (req), SYNTH_DIR (req), SYNTH_PER_SRC[8], STEPS[30], GUID[1.0]."""
import os, sys, glob
sys.path.insert(0, "/hpc/group/youlab/sa603/code/PixCell/controlnet")
import cv2, torch
from diffusers import DiffusionPipeline

CKPT = os.environ["PIXCELL_CN_CKPT"]
OUT = os.environ["SYNTH_DIR"]
REALS = "/hpc/group/youlab/sa603/data/multiplexed_eval/reals"
SRC_REPS = [8, 9, 10]
S = int(os.environ.get("SYNTH_PER_SRC", "8"))
STEPS = int(os.environ.get("STEPS", "30"))
GUID = float(os.environ.get("GUID", "1.0"))
os.makedirs(OUT, exist_ok=True)

pipe = DiffusionPipeline.from_pretrained(
    "StonyBrook-CVLab/PixCell-256-Cell-ControlNet",
    custom_pipeline="StonyBrook-CVLab/PixCell-pipeline-ControlNet",
    trust_remote_code=True, torch_dtype=torch.float32).to("cuda")
pipe.controlnet.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
pipe.controlnet.eval()
uncond = pipe.get_unconditional_embedding(1)

srcs = []
for rep in SRC_REPS:
    for fp in sorted(glob.glob(f"{REALS}/*_Rep{rep}.TIF")):
        srcs.append((os.path.basename(fp).split("_")[0], rep, fp))
print(f"[pixcell synth] {len(srcs)} sources x {S} -> {OUT} (steps={STEPS} guid={GUID})", flush=True)
for k, (cond, rep, fp) in enumerate(srcs):
    im = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (256, 256))
    for n in range(S):
        g = torch.Generator("cuda").manual_seed(729397049 + n)
        out = pipe(uni_embeds=uncond, negative_uni_embeds=uncond, controlnet_input=im,
                   guidance_scale=GUID, num_inference_steps=STEPS, num_images_per_prompt=1,
                   generator=g).images[0]
        out.save(f"{OUT}/{cond}_src{rep}_s{n}.png")
    if k % 30 == 0:
        print(f"  {k}/{len(srcs)}", flush=True)
print("[pixcell synth] done", flush=True)
