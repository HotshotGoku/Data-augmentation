"""Prepare data for the downstream-utility experiment (sim->exp), in the augmenter-native
BACKGROUND-BLACK space so real and synthetic targets align with no lossy conversion.

Selects N base colonies, writes real-only and real+synthetic training JSONs (absolute paths +
per-record `synthetic` flag), and generates M synthetic experimental replicates per colony with
the BASE (in-domain) augmenter (imports pipeline -> loads CKPT_PATH_V4 on the GPU).

Env: DS_N (colonies, default 50), DS_M (synth/colony, default 20),
     DS_OUT (staging dir), DS_SMOKE (1 = tiny 2x2 for validation).
Run on a GPU node. Upload to Data_augmentation repo root.
"""
import os
import glob
import json
import cv2
import numpy as np

import pipeline  # loads the BASE augmenter (CKPT_PATH_V4) on the GPU at import
import Data_augmentation.utils.shared_resources_config as _srcfg  # noqa: F401 (sets cldm path)
from cldm.preprocess import preprocess_experimental_backgroundblack
from cldm.config import SPECIFIC_FOLDER_SIM, SPECIFIC_FOLDER_EXP

DS_N = int(os.environ.get("DS_N", "50"))
DS_M = int(os.environ.get("DS_M", "20"))
DS_OUT = os.environ.get("DS_OUT", "/hpc/group/youlab/sa603/data/downstream_simexp")
if os.environ.get("DS_SMOKE", "0") == "1":
    DS_N, DS_M = 2, 2
    DS_OUT = DS_OUT + "_smoke"

SYNTH_DIR = os.path.join(DS_OUT, "synth")
os.makedirs(SYNTH_DIR, exist_ok=True)

# 1) Enumerate base colonies from the exp folder (filename = "<colony>_rot<deg>.TIF").
colony2files = {}
for fp in sorted(glob.glob(os.path.join(SPECIFIC_FOLDER_EXP, "*.TIF"))):
    fn = os.path.basename(fp)
    colony2files.setdefault(fn.split("_rot")[0], []).append(fn)
colonies = sorted(colony2files.keys())[:DS_N]
print(f"[prep] {len(colonies)}/{len(colony2files)} colonies, M={DS_M} synth each, out={DS_OUT}", flush=True)


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# 2) Real-only JSON (A): every rotation of the N colonies, sim+exp share the filename.
real_rows = []
for c in colonies:
    for fn in colony2files[c]:
        real_rows.append({"source_path": os.path.join(SPECIFIC_FOLDER_SIM, fn),
                          "target_path": os.path.join(SPECIFIC_FOLDER_EXP, fn),
                          "synthetic": 0, "prompt": ""})
dump(os.path.join(DS_OUT, "train_real.json"), real_rows)
print(f"[prep] train_real.json: {len(real_rows)} pairs", flush=True)

# 3) Generate M synthetic replicates per colony from its real exp (background-black source).
ARGS = {"prompt": "", "a_prompt": "",
        "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
        "num_samples": 4, "image_resolution": 256, "ddim_steps": 50, "guess_mode": False,
        "strength": 1.0, "scale": 9.0, "eta": 0.0}
synth_rows = []
for c in colonies:
    canon = colony2files[c][0]  # canonical (rot0.0) file, guaranteed to exist
    img = preprocess_experimental_backgroundblack(os.path.join(SPECIFIC_FOLDER_EXP, canon))
    made = 0
    seed = 1000
    while made < DS_M:
        outs = pipeline.process(img, **{**ARGS, "seed": seed})
        seed += 1
        for out in outs:
            if made >= DS_M:
                break
            name = f"{c}_synth{made}.png"
            cv2.imwrite(os.path.join(SYNTH_DIR, name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            synth_rows.append({"source_path": os.path.join(SPECIFIC_FOLDER_SIM, canon),
                               "target_path": os.path.join(SYNTH_DIR, name),
                               "synthetic": 1, "prompt": ""})
            made += 1
    print(f"[prep] {c}: {made} synth", flush=True)

dump(os.path.join(DS_OUT, "train_synth.json"), synth_rows)
dump(os.path.join(DS_OUT, "train_real_plus_synth.json"), real_rows + synth_rows)
print(f"[prep] train_synth.json: {len(synth_rows)} | train_real_plus_synth.json: {len(real_rows)+len(synth_rows)}", flush=True)
print("[prep] done", flush=True)
