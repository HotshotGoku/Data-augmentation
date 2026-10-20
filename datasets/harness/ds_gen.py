"""Generate N synth replicates per source image with an SD1.5 ControlNet ckpt (FT_EVAL_CKPT), named
<source_stem>_s{i}.png so ds_downstream.py can match each synth to its real source. Env:
FT_EVAL_CKPT, SOURCES (glob incl. extension), OUT, N [8], BATCH [4], STRENGTH [1.0], SCALE [9.0], ETA [0.0], SEED."""
import os, glob, cv2
assert os.environ.get("FT_EVAL_CKPT"), "set FT_EVAL_CKPT"
import pipeline  # loads FT_EVAL_CKPT on the GPU at import
SRC = os.environ["SOURCES"]; OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
N = int(os.environ.get("N", "8")); BATCH = int(os.environ.get("BATCH", "4"))
STR = float(os.environ.get("STRENGTH", "1.0")); SCALE = float(os.environ.get("SCALE", "9.0")); ETA = float(os.environ.get("ETA", "0.0"))
SEED0 = int(os.environ.get("SEED", "729397049"))
base = dict(prompt="", a_prompt="", n_prompt="longbody, lowres, bad anatomy, cropped, worst quality, low quality",
            image_resolution=256, ddim_steps=50, guess_mode=False, strength=STR, scale=SCALE, eta=ETA)
srcs = sorted(glob.glob(SRC)); print(f"[ds_gen] {len(srcs)} sources x {N} (strength={STR} scale={SCALE}) -> {OUT}", flush=True)
for fp in srcs:
    stem = os.path.splitext(os.path.basename(fp))[0]
    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    i = 0
    while i < N:
        b = min(BATCH, N - i)
        outs = pipeline.process(img, num_samples=b, seed=SEED0 + i, **base)
        for out in outs:
            i += 1; cv2.imwrite(os.path.join(OUT, f"{stem}_s{i}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("[ds_gen] done", flush=True)
