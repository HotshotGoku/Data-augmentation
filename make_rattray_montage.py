"""Rattray before/after montage: source | baseline | fine-tuned(best) | real sibling, from the
already-generated sweep-eval outputs. Run on a compute node."""
import os, glob, cv2, numpy as np
RAT = "/hpc/group/youlab/sa603/data/external_datasets/rattray_2023"
SW = "/hpc/group/youlab/sa603/code/Data_augmentation/rattray_sweep_eval"
BEST = "rat_none_1e-5_epoch=5"
TESTSRC, TESTREAL = f"{RAT}/test_sources", f"{RAT}/test_reals"

srcs = {os.path.basename(fp).split("_")[0]: fp for fp in sorted(glob.glob(f"{TESTSRC}/*.TIF"))}
strains = sorted(srcs)[:4]

def load(p, sz=256):
    if not p or not os.path.exists(p):
        return np.full((sz, sz, 3), 30, np.uint8)
    return cv2.resize(cv2.imread(p, cv2.IMREAD_COLOR), (sz, sz))

def lbl(im, t, col=(255, 255, 255)):
    c = im.copy(); cv2.rectangle(c, (0, 0), (256, 22), (15, 15, 15), -1)
    cv2.putText(c, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA); return c

rows = []
for s in strains:
    src_rep = os.path.basename(srcs[s]).split("_")[1].split(".")[0]
    reals = [f for f in sorted(glob.glob(f"{TESTREAL}/{s}_*.TIF")) if not f.endswith(f"{s}_{src_rep}.TIF")]
    cells = [lbl(load(srcs[s]), f"source {s}"),
             lbl(load(f"{SW}/baseline_base/gen/{s}_1.png"), "baseline", (165, 165, 252)),
             lbl(load(f"{SW}/{BEST}/gen/{s}_1.png"), "fine-tuned", (172, 239, 134)),
             lbl(load(reals[0] if reals else None), "real sibling")]
    sep = np.full((256, 4, 3), 30, np.uint8)
    rows.append(np.hstack([x for c in cells for x in (c, sep)]))
gap = np.full((6, rows[0].shape[1], 3), 0, np.uint8)
cv2.imwrite(f"{SW}/rattray_comparison.png", np.vstack([x for r in rows for x in (r, gap)]))
print("wrote rattray_comparison.png")
