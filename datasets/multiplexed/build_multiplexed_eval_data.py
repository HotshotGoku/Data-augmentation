"""Prepare Kinshuk's multiplexed-sensing data for a realism eval (test-only, 256x256).

Structure: files are Rep{X}_Row{Y}_Column{Z}.TIF. The CONDITION is Row_Column; its replicates
(Rep{X}) are spread across train/val/test. So we pool ALL reps (all 3 splits) as the sibling set,
and generate from one test-set rep per condition.

Writes (all 256x256, color, per Kinshuk's reptorep_infer_multiplexeddata.py conversion):
  reals/<cond>_Rep<X>.TIF   -- every replicate of every condition (train+val+test) -> realism pool
  sources/<cond>.TIF        -- one test-set rep per condition -> generation input
Condition token = R<Y>C<Z> (no underscore, so eval_metrics prefix grouping = condition).
Run on the DCC.
"""
import os, glob, re, cv2

BASE = "/hpc/group/youlab/ks723/storage/Exp_images/Multiplexed_patterning"
OUT = "/hpc/group/youlab/sa603/data/multiplexed_eval"
REALS, SOURCES = OUT + "/reals", OUT + "/sources"
os.makedirs(REALS, exist_ok=True); os.makedirs(SOURCES, exist_ok=True)

def cond_rep(fn):
    m = re.match(r"Rep(\d+)_Row(\d+)_Column(\d+)\.TIF$", fn)
    return (f"R{m.group(2)}C{m.group(3)}", m.group(1)) if m else (None, None)

def conv(src, dst):
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    cv2.imwrite(dst, cv2.resize(im, (256, 256)))

# reals: every replicate across all three splits, grouped by condition
nr = 0
for split in ("Combined_train_set", "Combined_validation_set", "Combined_test_set"):
    for fp in sorted(glob.glob(f"{BASE}/{split}/*.TIF")):
        cond, rep = cond_rep(os.path.basename(fp))
        if cond:
            conv(fp, f"{REALS}/{cond}_Rep{rep}.TIF"); nr += 1

# sources: one test-set rep per condition (the generation input)
seen = set(); ns = 0
for fp in sorted(glob.glob(f"{BASE}/Combined_test_set/*.TIF")):
    cond, _ = cond_rep(os.path.basename(fp))
    if cond and cond not in seen:
        conv(fp, f"{SOURCES}/{cond}.TIF"); seen.add(cond); ns += 1

print(f"reals: {nr} imgs across {len(set(os.path.basename(f).split('_')[0] for f in glob.glob(REALS+'/*.TIF')))} conditions; sources: {ns}")
