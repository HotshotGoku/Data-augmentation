"""Diversity-boosted SD1.5: generate from the in-domain SD1.5 branching ControlNet at variable
conditioning STRENGTH + guidance SCALE (lower = less source-anchoring = more diverse). Conditioned,
so eval_metrics scores per-source diversity/CMMD/realism properly. Reuses the WORKING cldm pipeline."""
import os, glob, cv2
assert os.environ.get("FT_EVAL_CKPT"), "set FT_EVAL_CKPT"
import pipeline
from cldm.preprocess import preprocess_experimental_backgroundblack
SOURCES = os.environ.get("SOURCES", "/hpc/group/youlab/sa603/data/branching_256/test/sources")
OUT = os.environ["GEN_OUT"]; N = int(os.environ.get("NUM_SAMPLES", "4"))
STR = float(os.environ.get("STRENGTH", "1.0")); SCALE = float(os.environ.get("SCALE", "9.0"))
ETA = float(os.environ.get("ETA", "0.0"))   # stochastic DDIM: diversity WITHOUT lowering conditioning fidelity
os.makedirs(OUT, exist_ok=True)
ARGS = dict(prompt="", a_prompt="", n_prompt="longbody, lowres, bad anatomy, cropped, worst quality, low quality",
            num_samples=N, image_resolution=256, ddim_steps=50, guess_mode=False,
            strength=STR, scale=SCALE, seed=int(os.environ.get("SEED", "729397049")), eta=ETA)
srcs = sorted(glob.glob(f"{SOURCES}/*.TIF"))
print(f"[divgen] strength={STR} scale={SCALE} N={N} -> {OUT}", flush=True)
for k, fp in enumerate(srcs):
    prefix = os.path.basename(fp).split("_")[0]
    img = preprocess_experimental_backgroundblack(fp)
    outs = pipeline.process(img, **ARGS)
    for i, out in enumerate(outs, start=1):
        cv2.imwrite(os.path.join(OUT, f"{prefix}_{i}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("[divgen] done", flush=True)
