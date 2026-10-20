"""Build the GENERALIST fine-tune dataset from internal Exp_images replicate data (Phase 2).
Canonicalizes each domain to 256px, groups by condition ({id}_{rep} -> cond token w/ NO underscore
so infer/eval prefix grouping works), holds out ~15% of conditions per domain for a clean eval,
and writes combined train pairs + per-domain test sources/reals. Run on the DCC (CPU node ok).

Outputs under data/generalist/:
  canon/<code>/<cond>_Rep<rep>.TIF     all reps, 256px
  test/<code>/sources/<cond>_src.TIF   one rep from each held-out condition
  test/<code>/reals/<cond>_Rep<rep>.TIF  all reps of held-out conditions (realism pool)
  train_generalist.json                all ordered replicate pairs from TRAIN conditions (+multiplexed)
"""
import os, glob, re, json, shutil, cv2

E = "/hpc/group/youlab/ks723/storage/Exp_images"
OUT = "/hpc/group/youlab/sa603/data/generalist"
CANON, TEST = OUT + "/canon", OUT + "/test"
# code (no underscore!), path, glob
DATASETS = [
    ("pakp", f"{E}/EmrahPaKp_dataset_renamed", "*.TIF"),
    ("kl2",  f"{E}/KL_automatedcrops_2species_filtered_renamed", "*.jpg"),
    ("ff",   f"{E}/Final_folder_uniform_fixedseed", "*.TIF"),
    ("nlev", f"{E}/NL_evolution_library_Image", "*.TIF"),
    ("selx", f"{E}/Selected_Exps", "*.TIF"),
]
HOLDOUT_EVERY = 7  # ~15% of conditions held out per domain

def rep_id(fn):
    m = re.match(r"(.+)_(\d+)\.(TIF|tif|jpg|jpeg|png)$", fn)
    return (m.group(1), int(m.group(2))) if m else (None, None)

def conv(src, dst):
    cv2.imwrite(dst, cv2.resize(cv2.imread(src, cv2.IMREAD_COLOR), (256, 256)))

pairs = []
summary = []
for code, path, pat in DATASETS:
    cdir = f"{CANON}/{code}"; os.makedirs(cdir, exist_ok=True)
    byid = {}
    for f in sorted(glob.glob(f"{path}/{pat}")):
        i, r = rep_id(os.path.basename(f))
        if i:
            byid.setdefault(i, []).append((r, f))
    ids = sorted(byid)
    n_tr = n_te = 0
    for idx, cid in enumerate(ids):
        reps = sorted(byid[cid])
        if len(reps) < 2:
            continue  # need >=2 reps to form a pair / eval
        cond = f"{code}c{idx}"
        canon_paths = []
        for r, f in reps:
            dst = f"{cdir}/{cond}_Rep{r}.TIF"; conv(f, dst); canon_paths.append((r, dst))
        held = (idx % HOLDOUT_EVERY == 0)
        if held:
            sdir, rdir = f"{TEST}/{code}/sources", f"{TEST}/{code}/reals"
            os.makedirs(sdir, exist_ok=True); os.makedirs(rdir, exist_ok=True)
            shutil.copy(canon_paths[0][1], f"{sdir}/{cond}_src.TIF")
            for r, dp in canon_paths:
                shutil.copy(dp, f"{rdir}/{cond}_Rep{r}.TIF")
            n_te += 1
        else:
            for r1, p1 in canon_paths:
                for r2, p2 in canon_paths:
                    if r1 != r2:
                        pairs.append({"source_path": p1, "target_path": p2, "prompt": "", "dataset": code})
            n_tr += 1
    summary.append((code, len(ids), n_tr, n_te))
    print(f"{code}: conditions={len(ids)} train={n_tr} heldout={n_te}", flush=True)

# include the already-canonical multiplexed train pairs (same replicate-to-replicate task)
mplex_json = "/hpc/group/youlab/sa603/data/multiplexed_eval/train_multiplexed.json"
n_mplex = 0
if os.path.exists(mplex_json):
    for line in open(mplex_json):
        d = json.loads(line); d["dataset"] = "mplex"; pairs.append(d); n_mplex += 1

os.makedirs(OUT, exist_ok=True)
with open(OUT + "/train_generalist.json", "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
print(f"\nTOTAL train pairs: {len(pairs)} (internal {len(pairs)-n_mplex} + multiplexed {n_mplex})", flush=True)
print("per-domain:", summary, flush=True)
