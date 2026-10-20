"""Build the BRANCHING ('ff' = Final_folder) replicate-pair dataset at a chosen resolution, from the
NATIVE 1001x1001 originals in the Exp_images trove (normally downsampled to 256). Same pairing +
holdout logic as build_generalist_pairs.py (replicate->replicate within a condition, HOLDOUT_EVERY=7,
conditions sorted so the split is deterministic), so branching_1024 and branching_256 share the EXACT
same conditions/pairs/split and differ ONLY in resolution -- a clean isolation of the resolution effect
for PixCell-1024 vs PixCell-256.

Env: RES[1024]. Outputs under data/branching_<RES>/:
  canon/<cond>_Rep<r>.TIF          all reps, RESxRES
  test/sources/<cond>_src.TIF      one rep per held-out condition (eval sources)
  test/reals/<cond>_Rep<r>.TIF     all reps of held-out conditions (realism pool)
  train_branching.json             all ordered replicate pairs from TRAIN conditions
Run on a CPU compute node (dcc-agent). Set OPENCV_LOG_LEVEL=OFF to mute TIFF tag warnings."""
import os, glob, re, json, shutil, cv2

E = "/hpc/group/youlab/ks723/storage/Exp_images"
SRC = f"{E}/Final_folder_uniform_fixedseed"   # native 1001x1001 branching originals
RES = int(os.environ.get("RES", "1024"))
OUT = f"/hpc/group/youlab/sa603/data/branching_{RES}"
CANON, TEST = OUT + "/canon", OUT + "/test"
HOLDOUT_EVERY = 7
os.makedirs(CANON, exist_ok=True)

def rep_id(fn):
    m = re.match(r"(.+)_(\d+)\.(TIF|tif|jpg|jpeg|png)$", fn)
    return (m.group(1), int(m.group(2))) if m else (None, None)

def conv(src, dst):
    cv2.imwrite(dst, cv2.resize(cv2.imread(src, cv2.IMREAD_COLOR), (RES, RES)))

byid = {}
for f in sorted(glob.glob(f"{SRC}/*.TIF")):
    i, r = rep_id(os.path.basename(f))
    if i:
        byid.setdefault(i, []).append((r, f))
ids = sorted(byid)
pairs = []
n_tr = n_te = 0
for idx, cid in enumerate(ids):
    reps = sorted(byid[cid])
    if len(reps) < 2:
        continue  # need >=2 reps to form a pair / eval
    cond = f"ffc{idx}"
    cps = []
    for r, f in reps:
        dst = f"{CANON}/{cond}_Rep{r}.TIF"; conv(f, dst); cps.append((r, dst))
    if idx % HOLDOUT_EVERY == 0:
        sdir, rdir = f"{TEST}/sources", f"{TEST}/reals"
        os.makedirs(sdir, exist_ok=True); os.makedirs(rdir, exist_ok=True)
        shutil.copy(cps[0][1], f"{sdir}/{cond}_src.TIF")
        for r, dp in cps:
            shutil.copy(dp, f"{rdir}/{cond}_Rep{r}.TIF")
        n_te += 1
    else:
        for r1, p1 in cps:
            for r2, p2 in cps:
                if r1 != r2:
                    pairs.append({"source_path": p1, "target_path": p2, "prompt": "", "dataset": "ff"})
        n_tr += 1

with open(OUT + "/train_branching.json", "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
print(f"branching RES={RES}: conditions={len(ids)} train_conds={n_tr} heldout={n_te} pairs={len(pairs)}", flush=True)
print(f"out: {OUT}", flush=True)
