"""Build Rattray fine-tune data:
  - canonical 256x256 3-channel images (Rattray PNGs are already background-removed grayscale ~256px)
  - all within-strain ORDERED replicate pairs (source != target) for training
  - a ZERO-LEAKAGE held-out test split: whole strains reserved (every 6th) -> test_sources/ + test_reals/

Naming in: NoBackgroundPIL<strain>-<rep>.png. Run on a compute node. Writes under
/hpc/group/youlab/sa603/data/external_datasets/rattray_2023/.
"""
import os, glob, json, re, cv2

BASE = "/hpc/group/youlab/sa603/data/external_datasets/rattray_2023"
RAW = os.path.join(BASE, "images")
PROC = os.path.join(BASE, "processed")           # canonical train images
TESTSRC = os.path.join(BASE, "test_sources")     # 1 source per held-out strain (for generation)
TESTREAL = os.path.join(BASE, "test_reals")      # all replicates of held-out strains (for realism_vs_sibling)
for d in (PROC, TESTSRC, TESTREAL):
    os.makedirs(d, exist_ok=True)


def canon(path):
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    a = cv2.resize(a, (256, 256))
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)  # write as 3ch BGR


# group replicates by strain
strains = {}
for fp in sorted(glob.glob(os.path.join(RAW, "*.png"))):
    m = re.match(r"NoBackgroundPIL(\d+)-(\d+)\.png", os.path.basename(fp))
    if m:
        strains.setdefault(m.group(1), []).append((m.group(2), fp))

alls = sorted(strains)
test_s = set(alls[::6])                    # hold out every 6th strain (~13 of 77), zero leakage
train_s = [s for s in alls if s not in test_s]
print(f"{len(alls)} strains -> {len(train_s)} train, {len(test_s)} test", flush=True)

# TRAIN: canonical images + all ordered replicate pairs
rows = []
for s in train_s:
    paths = {}
    for r, fp in strains[s]:
        outp = os.path.join(PROC, f"PIL{s}-{r}.png"); cv2.imwrite(outp, canon(fp)); paths[r] = outp
    reps = list(paths)
    for i in reps:
        for j in reps:
            if i != j:
                rows.append({"source_path": paths[i], "target_path": paths[j], "prompt": ""})
with open(os.path.join(BASE, "train_rattray.json"), "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print(f"train pairs: {len(rows)} (from {len(train_s)} strains)", flush=True)

# TEST: all replicates as reals (grouped by strain prefix); first replicate as the generation source
ns = 0
for s in test_s:
    reps = sorted(strains[s])
    for r, fp in reps:
        cv2.imwrite(os.path.join(TESTREAL, f"{s}_{r}.TIF"), canon(fp)); ns += 1
    r0, fp0 = reps[0]
    cv2.imwrite(os.path.join(TESTSRC, f"{s}_{r0}.TIF"), canon(fp0))
print(f"test: {len(test_s)} source strains, {ns} real replicate imgs", flush=True)
print("[build_rattray] done", flush=True)
