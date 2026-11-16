"""Build Rattray fine-tune data:
  - canonical 256x256 3-channel images (Rattray PNGs are already background-removed grayscale ~256px)
  - all within-strain ORDERED replicate pairs (source != target) for training
  - a ZERO-LEAKAGE held-out test split: whole strains reserved (every 6th) -> test_sources/ + test_reals/

Naming in: NoBackgroundPIL<strain>-<rep>.png. Run on a compute node. Writes under
/hpc/group/youlab/sa603/data/external_datasets/rattray_2023/.
"""
import os, glob, json, re, cv2
import numpy as np

BASE = "/hpc/group/youlab/sa603/data/external_datasets/rattray_2023"
RAW = os.path.join(BASE, "images")
# OUT_SUFFIX="_pp" -> inscribed build, written alongside the legacy one (non-destructive).
SUF = os.environ.get("OUT_SUFFIX", "")
PROC = os.path.join(BASE, "processed" + SUF)       # canonical train images
TESTSRC = os.path.join(BASE, "test_sources" + SUF) # 1 source per held-out strain (for generation)
TESTREAL = os.path.join(BASE, "test_reals" + SUF)  # all replicates of held-out strains (for realism_vs_sibling)
# Kinshuk 2026-11-17: scale content so the max colony radius (~148) maps to the rotation-safe radius
# (128 = frame half-width) -> fine rotation never pushes edge/strain morphology out of the 256^2 frame.
# ON for the _pp build, OFF (1.0 no-op) for a legacy rebuild so default output stays byte-identical.
INSCRIBE = float(os.environ.get("INSCRIBE", "0.865" if SUF else "1.0"))
for d in (PROC, TESTSRC, TESTREAL):
    os.makedirs(d, exist_ok=True)


def inscribe(img, frac=INSCRIBE, fill=255):
    """Scale content to frac, center on a fill (white) canvas -> rotation-safe margin."""
    if frac >= 1.0:
        return img
    h, w = img.shape[:2]
    sh, sw = int(round(h * frac)), int(round(w * frac))
    s = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), fill, np.uint8)
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    canvas[y0:y0 + sh, x0:x0 + sw] = s
    return canvas


def canon(path):
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    a = cv2.resize(a, (256, 256))
    return inscribe(cv2.cvtColor(a, cv2.COLOR_GRAY2BGR))  # 3ch BGR, 0.865 inscribe (white fill) for _pp


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
with open(os.path.join(BASE, "train_rattray" + SUF + ".json"), "w") as f:
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
