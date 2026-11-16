"""Build a STANDALONE SwarmEvo generator-training set (decoupled from the rattray two-stage).
For each colony's time-lapse, bakes OUR dish_remove (rim0.20, BLACK exterior) to disk and pairs
(frame_i, frame_i+step) growth deltas. The dish mask confines content to a centered disc (radius
~0.80*128 ~= 102 < 128) so runtime fine rotation on black fill is corner-artifact-free (no inscribe
needed, unlike rattray). Trains on the MAIN loader (reptorep_dataset.py) with GEN_AUG.

Env: PAIRS_PER_COLONY[20]. Output: swarmevo/train_swarmevo_pp.json + swarmevo/canon_pp/<colony>/.
Run on a CPU compute node (dcc-agent).
"""
import os, glob, json, sys
import cv2

# repo/pipeline/postprocess on path for dish_remove (+ its dish_normalize sibling)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "pipeline", "postprocess"))
from dish_remove import remove_dish

BASE = "/hpc/group/youlab/sa603/data/external_datasets/swarmevo"
CANON = BASE + "/canon"
PROC = BASE + "/canon_pp"                 # dish-removed frames (black exterior)
OUT = BASE + "/train_swarmevo_pp.json"
PAIRS_PER_COLONY = int(os.environ.get("PAIRS_PER_COLONY", "20"))
os.makedirs(PROC, exist_ok=True)


def bake(fp, out):
    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    removed, _ = remove_dish(img)                    # rim0.20, black exterior
    cv2.imwrite(out, cv2.cvtColor(removed, cv2.COLOR_RGB2BGR))


pairs, ncol = [], 0
for cd in sorted(glob.glob(CANON + "/*/")):
    name = os.path.basename(cd.rstrip("/"))
    frames = sorted(glob.glob(cd + "*.png") + glob.glob(cd + "*.jpg"))
    if len(frames) < 3:
        continue
    step = max(1, len(frames) // (PAIRS_PER_COLONY + 1))
    idxs = list(range(0, len(frames) - step, step))
    pdir = os.path.join(PROC, name); os.makedirs(pdir, exist_ok=True)
    baked = {}
    for i in sorted(set(idxs) | {i + step for i in idxs}):
        o = os.path.join(pdir, os.path.basename(frames[i])); bake(frames[i], o); baked[i] = o
    for i in idxs:
        pairs.append({"source_path": baked[i], "target_path": baked[i + step], "prompt": ""})
    ncol += 1

with open(OUT, "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
print(f"[swarmevo-pp] {len(pairs)} pairs from {ncol} colonies (PAIRS_PER_COLONY={PAIRS_PER_COLONY}) -> {OUT}", flush=True)
