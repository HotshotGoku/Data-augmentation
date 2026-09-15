"""Build JOINT broadening train set = Rattray replicate pairs + SwarmEvo time-lapse pairs.
SwarmEvo (Enterobacter swarming, whole-plate) is on-modality (bacterial colonies on a plate) — a
much closer broadening match to Rattray than Myxococcus. Consecutive frames within a colony form
growth pairs. Builds canonical 256px 3ch pairs, balances to ~Rattray count, writes
train_joint_rattray_swarmevo.json. Run on a compute node.
"""
import os, glob, json, re, random
import cv2

BASE = "/hpc/group/youlab/sa603/data/external_datasets"
SWARM = os.path.join(BASE, "swarmevo", "prediction", "raw_imgs")
PROC = os.path.join(BASE, "swarmevo", "processed")
RAT_JSON = os.path.join(BASE, "rattray_2023", "train_rattray.json")
os.makedirs(PROC, exist_ok=True)
random.seed(42)

def natkey(p):
    ns = re.findall(r"\d+", os.path.basename(p))
    return [int(n) for n in ns] if ns else [0]

def canon(fp, out):
    im = cv2.imread(fp, cv2.IMREAD_COLOR)
    cv2.imwrite(out, cv2.resize(im, (256, 256)))

# each colony dir under train/ and test/ has a time-series of frames
colony_dirs = []
for split in ("train", "test"):
    d = os.path.join(SWARM, split)
    if os.path.isdir(d):
        for c in sorted(os.listdir(d)):
            cd = os.path.join(d, c)
            if os.path.isdir(cd):
                colony_dirs.append(cd)
print(f"[swarmevo] {len(colony_dirs)} colony dirs", flush=True)

PAIRS_PER_COLONY = 20   # spread across each time-series
pairs, k = [], 0
for cd in colony_dirs:
    frames = sorted(glob.glob(os.path.join(cd, "*.jpg")) + glob.glob(os.path.join(cd, "*.png")), key=natkey)
    if len(frames) < 3:
        continue
    step = max(1, len(frames) // (PAIRS_PER_COLONY + 1))
    for i in range(0, len(frames) - step, step):
        s_out = os.path.join(PROC, f"sw{k}_s.png"); t_out = os.path.join(PROC, f"sw{k}_t.png")
        canon(frames[i], s_out); canon(frames[i + step], t_out)
        pairs.append({"source_path": s_out, "target_path": t_out, "prompt": ""}); k += 1

rat_pairs = [json.loads(l) for l in open(RAT_JSON)]
random.shuffle(pairs)
pairs = pairs[:len(rat_pairs)]   # balance the joint set ~1:1
print(f"[swarmevo] {len(pairs)} swarm pairs (balanced to {len(rat_pairs)} rattray)", flush=True)

joint = rat_pairs + pairs
random.shuffle(joint)
outp = os.path.join(BASE, "rattray_2023", "train_joint_rattray_swarmevo.json")
with open(outp, "w") as f:
    for r in joint:
        f.write(json.dumps(r) + "\n")
print(f"[joint] {len(joint)} pairs -> {outp}", flush=True)
