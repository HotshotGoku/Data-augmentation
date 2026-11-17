"""Build a CLEAN standalone kl (2-species) replicate-pair dataset for the runtime-aug approach.
Replaces the old train_2sp.json, which pointed at KL_..._AUG100 (pre-baked 100x rotation on disk) --
the disk-bake we're moving away from. kl is FLUORESCENT (label = species ratio in color/intensity),
so NO dish preprocess and NO photometric aug: just resize to 256 + replicate pairs + holdout. Runtime
rotate/flip (brightness OFF) is applied by the GEN_AUG dataloader.

Outputs {source,target,prompt,folder} JSONL (drop-in for reptorep_dataset.MyDataset -- no cldm adapter
needed, matching the old train_2sp.json format). Run on a CPU compute node (dcc-agent).
Outputs under data/kl_256_pp/:
  canon/<cond>_Rep<r>.TIF     all reps, 256px fluor (color kept)
  test/sources, test/reals    held-out conditions (eval)
  train_kl.json               ordered replicate pairs from TRAIN conditions
"""
import os, glob, re, json, shutil, cv2

SRC = "/hpc/group/youlab/ks723/storage/Exp_images/KL_automatedcrops_2species_filtered_renamed"  # non-AUG100
OUT = "/hpc/group/youlab/sa603/data/kl_256_pp"
CANON, TEST = OUT + "/canon", OUT + "/test"
HOLDOUT_EVERY = 7
os.makedirs(CANON, exist_ok=True)


def rep_id(fn):
    m = re.match(r"(.+)_(\d+)\.(TIF|tif|jpg|jpeg|png)$", fn)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def conv(src, dst):
    cv2.imwrite(dst, cv2.resize(cv2.imread(src, cv2.IMREAD_COLOR), (256, 256)))  # fluor: resize only, keep color


byid = {}
for f in sorted(glob.glob(f"{SRC}/*.jpg")):
    i, r = rep_id(os.path.basename(f))
    if i:
        byid.setdefault(i, []).append((r, f))
ids = sorted(byid)
pairs = []
n_tr = n_te = 0
for idx, cid in enumerate(ids):
    reps = sorted(byid[cid])
    if len(reps) < 2:
        continue
    cond = f"klc{idx}"
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
                    pairs.append({"source": os.path.basename(p1), "target": os.path.basename(p2),
                                  "prompt": "", "folder": CANON})
        n_tr += 1

with open(OUT + "/train_kl.json", "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
print(f"kl: conditions={len(ids)} train_conds={n_tr} heldout={n_te} pairs={len(pairs)}", flush=True)
print(f"out: {OUT}", flush=True)
