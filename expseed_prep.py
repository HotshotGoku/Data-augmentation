"""Prepare data for the exp->seed downstream-utility experiment (replicate-boost with OUR augmenter).

The exp->seed inverse model maps an experimental colony image -> 32x32 seed-location map. Many
replicates of a colony legitimately share ONE seed, so adding augmenter replicates is
correspondence-preserving (unlike the sim->exp test). CRITICAL prerequisite: the augmenter must
preserve colony/seed LOCATIONS in its replicates — this script's smoke mode is for verifying that.

Builds parallel source/target folders (aligned by identical zero-padded filenames, so the dataset's
natsorted pairing lines up exp[i] <-> seed[i]):
  A (baseline):        1 existing replicate per seed
  B (replicate-boost): 1 existing + DS_M augmenter replicates per seed (each paired with the SAME seed)

Env: ES_N (seeds, default 2500), ES_M (augmenter reps/seed, default 2), ES_OUT, ES_SMOKE(1 -> 3 seeds x2).
Run on a GPU node (imports pipeline -> base augmenter on GPU). Upload to Data_augmentation root.
"""
import os
import glob
import shutil
import cv2

import pipeline  # base (in-domain) augmenter, CKPT_PATH_V4, on GPU at import
import Data_augmentation.utils.shared_resources_config as _srcfg  # noqa: F401
from cldm.preprocess import preprocess_experimental_backgroundblack

EXIST_EXP = "/hpc/group/youlab/ks723/storage/Physics_constrained_DL_pattern_prediction/inference/v202642_19_SIMTOEXP"
EXIST_SEED = "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_050924/Sim_output_32x32_3reps"
ES_N = int(os.environ.get("ES_N", "2500"))
ES_M = int(os.environ.get("ES_M", "2"))
ES_OUT = os.environ.get("ES_OUT", "/hpc/group/youlab/sa603/data/downstream_expseed")
if os.environ.get("ES_SMOKE", "0") == "1":
    ES_N, ES_M, ES_OUT = 3, 2, ES_OUT + "_smoke"

# Existing files look like Output_<seedid>_<rep>.png  <->  Input_<seedid>_<rep>.png ; rep 1 is canonical.
seeds = []
for fp in sorted(glob.glob(os.path.join(EXIST_EXP, "Output_*_1.png"))):
    sid = os.path.basename(fp)[len("Output_"):-len("_1.png")]
    if os.path.exists(os.path.join(EXIST_SEED, f"Input_{sid}_1.png")):
        seeds.append(sid)
seeds = seeds[:ES_N]
print(f"[esprep] {len(seeds)} seeds, M={ES_M} augmenter reps each, out={ES_OUT}", flush=True)

dirs = {d: os.path.join(ES_OUT, d) for d in ("A_exp", "A_seed", "B_exp", "B_seed")}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

ARGS = {"prompt": "", "a_prompt": "",
        "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
        "num_samples": 4, "image_resolution": 256, "ddim_steps": 50, "guess_mode": False,
        "strength": 1.0, "scale": 9.0, "eta": 0.0}

for j, sid in enumerate(seeds):
    tag = f"{j:05d}"
    exp1 = os.path.join(EXIST_EXP, f"Output_{sid}_1.png")
    seed1 = os.path.join(EXIST_SEED, f"Input_{sid}_1.png")
    # A + B share the canonical real replicate (r0)
    for d in ("A_exp", "B_exp"):
        shutil.copy(exp1, os.path.join(dirs[d], f"{tag}_r0.png"))
    for d in ("A_seed", "B_seed"):
        shutil.copy(seed1, os.path.join(dirs[d], f"{tag}_r0.png"))
    # B: DS_M augmenter replicates from the canonical exp, each paired with the SAME seed
    img = preprocess_experimental_backgroundblack(exp1)
    made, seed_n = 0, 1000
    while made < ES_M:
        outs = pipeline.process(img, **{**ARGS, "seed": seed_n}); seed_n += 1
        for out in outs:
            if made >= ES_M:
                break
            cv2.imwrite(os.path.join(dirs["B_exp"], f"{tag}_r{made+1}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            shutil.copy(seed1, os.path.join(dirs["B_seed"], f"{tag}_r{made+1}.png"))
            made += 1
    if (j + 1) % 100 == 0 or os.environ.get("ES_SMOKE", "0") == "1":
        print(f"[esprep] {j+1}/{len(seeds)} ({sid})", flush=True)

nA = len(glob.glob(os.path.join(dirs["A_exp"], "*.png")))
nB = len(glob.glob(os.path.join(dirs["B_exp"], "*.png")))
print(f"[esprep] A pairs={nA}  B pairs={nB}  (B/A={nB/max(nA,1):.1f}x)", flush=True)
print("[esprep] done", flush=True)
