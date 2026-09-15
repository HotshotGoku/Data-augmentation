"""Generate synthetic multiplexed replicates from the fine-tuned augmenter, for the downstream
utility experiment. Sources = train reps 8,9,10 (present in all 70 conditions); SYNTH_PER_SRC each.
Set SYNTH_CKPT to the mplex-ft epoch2 ckpt. Outputs synth/<cond>_src<rep>_s<n>.png. Run on DCC GPU."""
import os
os.environ.setdefault("FT_EVAL_CKPT", os.environ["SYNTH_CKPT"])  # pipeline loads this at import
import glob, cv2
import pipeline
import Data_augmentation.utils.shared_resources_config as _srcfg  # noqa: F401

REALS = "/hpc/group/youlab/sa603/data/multiplexed_eval/reals"
OUT = os.environ.get("SYNTH_DIR", "/hpc/group/youlab/sa603/data/multiplexed_eval/synth_ft")
SRC_REPS = [8, 9, 10]
S = int(os.environ.get("SYNTH_PER_SRC", "8"))
os.makedirs(OUT, exist_ok=True)

BASE = {"prompt": "", "a_prompt": "",
        "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
        "image_resolution": 256, "ddim_steps": 50, "guess_mode": False,
        "strength": 1.0, "scale": 9.0, "eta": 0.0}
CHUNK = 4  # memory-safe; distinct seed per chunk so the S samples aren't duplicates

srcs = []
for rep in SRC_REPS:
    for fp in sorted(glob.glob(f"{REALS}/*_Rep{rep}.TIF")):
        srcs.append((os.path.basename(fp).split("_")[0], rep, fp))
print(f"[gen_synth] {len(srcs)} sources x {S} samples -> {OUT}", flush=True)

for k, (cond, rep, fp) in enumerate(srcs):
    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    n_done = 0
    for ci in range((S + CHUNK - 1) // CHUNK):
        a = dict(BASE); a["num_samples"] = min(CHUNK, S - n_done); a["seed"] = 729397049 + ci
        for out in pipeline.process(img, **a):
            cv2.imwrite(f"{OUT}/{cond}_src{rep}_s{n_done}.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            n_done += 1
    if k % 20 == 0:
        print(f"  {k}/{len(srcs)}", flush=True)
print("[gen_synth] done", flush=True)
