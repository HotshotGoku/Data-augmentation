"""Generate synthetic branching replicates from the in-domain augmenter for the colony-feature
downstream experiment. Sources = every real replicate in REAL_DIR whose condition is NOT in
EXCLUDE_DIR (so we synth the TRAIN pool and never the held-out TEST conditions). SYNTH_PER_SRC
seeded samples each -> synth/<cond>_<tag>_s<n>.png. Mirrors gen_multiplexed_synth.py.
Env: SYNTH_CKPT (req), SYNTH_DIR, REAL_DIR [canon], EXCLUDE_DIR [test/reals], SYNTH_PER_SRC[8]."""
import os
os.environ.setdefault("FT_EVAL_CKPT", os.environ["SYNTH_CKPT"])  # pipeline loads this at import
import glob, cv2
import pipeline
import Data_augmentation.utils.shared_resources_config as _srcfg  # noqa: F401

REAL_DIR = os.environ.get("REAL_DIR", "/hpc/group/youlab/sa603/data/branching_256/canon")
EXCLUDE_DIR = os.environ.get("EXCLUDE_DIR", "/hpc/group/youlab/sa603/data/branching_256/test/reals")
OUT = os.environ.get("SYNTH_DIR", "/hpc/group/youlab/sa603/data/branching_downstream/synth_train_generalist")
S = int(os.environ.get("SYNTH_PER_SRC", "8"))
os.makedirs(OUT, exist_ok=True)

def cond_of(p): return os.path.basename(p).split("_")[0]
exclude = {cond_of(p) for p in glob.glob(f"{EXCLUDE_DIR}/*.TIF")}
BASE = {"prompt": "", "a_prompt": "",
        "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
        "image_resolution": 256, "ddim_steps": 50, "guess_mode": False,
        "strength": 1.0, "scale": 9.0, "eta": 0.0}
CHUNK = 4

srcs = []
for fp in sorted(glob.glob(f"{REAL_DIR}/*_Rep*.TIF") + glob.glob(f"{REAL_DIR}/*_src.TIF")):
    c = cond_of(fp)
    if c in exclude:
        continue
    b = os.path.basename(fp).replace(".TIF", "")
    tag = b.split("_", 1)[1] if "_" in b else "src"
    srcs.append((c, tag, fp))
print(f"[gen_branch] {len(srcs)} train-pool sources (excluded {len(exclude)} test conds) x {S} -> {OUT}", flush=True)

for k, (cond, tag, fp) in enumerate(srcs):
    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    n_done = 0
    for ci in range((S + CHUNK - 1) // CHUNK):
        a = dict(BASE); a["num_samples"] = min(CHUNK, S - n_done); a["seed"] = 729397049 + ci
        for out in pipeline.process(img, **a):
            cv2.imwrite(f"{OUT}/{cond}_{tag}_s{n_done}.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            n_done += 1
    if k % 40 == 0:
        print(f"  {k}/{len(srcs)}", flush=True)
print("[gen_branch] done", flush=True)
