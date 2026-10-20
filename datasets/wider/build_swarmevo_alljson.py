"""Build the FULL SwarmEvo pair JSON from ALL canonicalized frames (no subsampling).
For each colony, pairs (frame_i, frame_i+DELTA) for every i -> uses every frame as a source, with a
DELTA-step growth delta. Run on the DCC after the canon dir is uploaded.
"""
import os, glob, json
CANON = "/hpc/group/youlab/sa603/data/external_datasets/swarmevo/canon"
OUT = "/hpc/group/youlab/sa603/data/external_datasets/swarmevo/train_swarmevo_all.json"
DELTA = int(os.environ.get("SW_DELTA", "30"))

pairs, ncol = [], 0
for cd in sorted(glob.glob(CANON + "/*/")):
    frames = sorted(glob.glob(cd + "*.png"))
    if len(frames) < DELTA + 2:
        continue
    ncol += 1
    for i in range(0, len(frames) - DELTA):
        pairs.append({"source_path": frames[i], "target_path": frames[i + DELTA], "prompt": ""})
with open(OUT, "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
print(f"{len(pairs)} pairs from {ncol} colonies (DELTA={DELTA}) -> {OUT}")
