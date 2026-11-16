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
import os, glob, re, json, shutil, cv2, numpy as np

E = "/hpc/group/youlab/ks723/storage/Exp_images"
SRC = f"{E}/Final_folder_uniform_fixedseed"   # native 1001x1001 branching originals
RES = int(os.environ.get("RES", "1024"))
# OUT_SUFFIX="_pp" -> initialstage_mod-preprocessed build, written alongside the legacy one (non-destructive).
OUT = f"/hpc/group/youlab/sa603/data/branching_{RES}" + os.environ.get("OUT_SUFFIX", "")
CANON, TEST = OUT + "/canon", OUT + "/test"
HOLDOUT_EVERY = 7
os.makedirs(CANON, exist_ok=True)

def rep_id(fn):
    m = re.match(r"(.+)_(\d+)\.(TIF|tif|jpg|jpeg|png)$", fn)
    return (m.group(1), int(m.group(2))) if m else (None, None)

# Kinshuk's preprocess_experimental_initialstage_mod, inlined verbatim (cv2+numpy; his cldm module needs
# torch) from Simulation_templated_pattern_prediction/.../cldm/preprocess.py:503 -- Hough-detect the plate,
# mask radius-82, contrast a1.5/b50, resize, WHITE exterior. Returns RGB, or None if no circle found.
def initialstage_mod(img_path, size):
    img = cv2.imread(img_path)
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=20, maxRadius=600)
    if circles is None:
        return None
    x, y, r = np.round(circles[0, 0]).astype(int)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(m, (int(x), int(y)), int(r - 82), (255, 255, 255), -1)
    adj = cv2.convertScaleAbs(cv2.bitwise_and(img, img, mask=m), alpha=1.5, beta=50)
    rgb = cv2.resize(cv2.cvtColor(adj, cv2.COLOR_BGR2RGB), (size, size))
    mr = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.bitwise_and(rgb, rgb, mask=mr)
    rgb[mr == 0] = [255, 255, 255]
    return rgb


def conv(src, dst):
    out = initialstage_mod(src, RES)                       # dish-cropped, white exterior, RGB
    if out is None:                                        # no circle found -> plain resize fallback
        out = cv2.cvtColor(cv2.resize(cv2.imread(src, cv2.IMREAD_COLOR), (RES, RES)), cv2.COLOR_BGR2RGB)
    cv2.imwrite(dst, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))  # cv2 writes BGR; file ends up standard RGB

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
