"""Check #1: generate from the retrained in-domain SD1.5 ControlNet on the 28 held-out test sources,
matching the Flux eval contract (NUM_SAMPLES per source -> {prefix}_{i}.png) so eval_metrics.py scores
realism/CMMD/diversity apples-to-apples vs test/reals. pipeline.py loads the ckpt from env FT_EVAL_CKPT."""
import os, glob, cv2
assert os.environ.get("FT_EVAL_CKPT"), "set FT_EVAL_CKPT to the branching SD1.5 checkpoint"
import pipeline  # loads ControlLDM.load_from_checkpoint(CKPT_PATH_V4=FT_EVAL_CKPT).cuda() at import
from cldm.preprocess import preprocess_experimental_backgroundblack

SOURCES = os.environ.get("SOURCES", "/hpc/group/youlab/sa603/data/branching_256/test/sources")
OUT = os.environ["GEN_OUT"]; N = int(os.environ.get("NUM_SAMPLES", "4"))
os.makedirs(OUT, exist_ok=True)
ARGS = dict(prompt="", a_prompt="",
            n_prompt="longbody, lowres, bad anatomy, cropped, worst quality, low quality",
            num_samples=N, image_resolution=256, ddim_steps=50, guess_mode=False,
            strength=1.0, scale=9.0, seed=int(os.environ.get("SEED", "729397049")), eta=0.0)
srcs = sorted(glob.glob(f"{SOURCES}/*.TIF"))
print(f"[sd15-gen] ckpt={os.environ['FT_EVAL_CKPT']}\n[sd15-gen] {len(srcs)} sources x {N} -> {OUT}", flush=True)
for k, fp in enumerate(srcs):
    prefix = os.path.basename(fp).split("_")[0]
    img = preprocess_experimental_backgroundblack(fp)
    outs = pipeline.process(img, **ARGS)
    for i, out in enumerate(outs, start=1):
        cv2.imwrite(os.path.join(OUT, f"{prefix}_{i}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    if k % 5 == 0:
        print(f"  {k}/{len(srcs)} {prefix}", flush=True)
print("[sd15-gen] done", flush=True)
