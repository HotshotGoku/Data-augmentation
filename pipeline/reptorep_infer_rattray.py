"""Generate replicates from Rattray test-source images (already canonical 256px 3ch — no preprocess).
Loads the checkpoint via pipeline (FT_EVAL_CKPT env override of CKPT_PATH_V4). Mirrors
reptorep_infer_ood.py but skips preprocessing since the inputs are already processed.

  python reptorep_infer_rattray.py --input_dir <test_sources> --out_dir <gen>
Upload to Data_augmentation repo root.
"""
import os, glob, cv2, argparse
import pipeline  # loads FT_EVAL_CKPT (or CKPT_PATH_V4) on the GPU at import
import Data_augmentation.utils.shared_resources_config as _srcfg  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", required=True)
ap.add_argument("--out_dir", required=True)
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# Inference knobs are env-overridable (for the Supp-Fig-16 knob sweep); defaults = paper defaults.
ARGS = {"prompt": "", "a_prompt": "",
        "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
        "num_samples": 2, "image_resolution": 256,
        "ddim_steps": int(os.environ.get("INFER_DDIM", "50")),
        "guess_mode": os.environ.get("INFER_GUESS", "0") == "1",
        "strength": float(os.environ.get("INFER_STRENGTH", "1.0")),
        "scale": float(os.environ.get("INFER_SCALE", "9.0")),
        "seed": 729397049, "eta": 0.0}

prefix_map = {}
for fp in sorted(glob.glob(os.path.join(args.input_dir, "*.TIF"))):
    prefix_map.setdefault(os.path.basename(fp).split("_")[0], fp)

print(f"{len(prefix_map)} test sources from {args.input_dir}", flush=True)
for prefix, fp in prefix_map.items():
    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  # already canonical
    outs = pipeline.process(img, **ARGS)
    for i, out in enumerate(outs, start=1):
        cv2.imwrite(os.path.join(args.out_dir, f"{prefix}_{i}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("[infer_rattray] done", flush=True)
