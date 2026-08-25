"""
Single-folder OOD inference for the reptorep augmenter (variant of reptorep_infer.py).

Generates num_samples replicates per numeric prefix from ONE input folder and saves {prefix}_{i}.png.
Used to test the model on a held-out SPECIES (e.g. the KL two-species fluorescence set) that was
never in training — the out-of-distribution probe.

Usage:
  python reptorep_infer_ood.py --input_dir <folder of *.TIF> --out_dir <writable folder>
"""
import os, glob, cv2, argparse

import pipeline  # loads the trained model (CKPT_PATH_V4) onto the GPU at import
import Data_augmentation.utils.shared_resources_config as shared_resources_config  # noqa: F401
from cldm.preprocess import preprocess_experimental_backgroundblack

ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", required=True)
ap.add_argument("--out_dir", required=True)
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

ARGS = {
    "prompt": "", "a_prompt": "",
    "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
    "num_samples": 2, "image_resolution": 256, "ddim_steps": 50,
    "guess_mode": False, "strength": 1.0, "scale": 9.0, "seed": 729397049, "eta": 0.0,
}

# one source file per numeric prefix (matches reptorep_infer.py)
prefix_map = {}
for fp in sorted(glob.glob(os.path.join(args.input_dir, "*.TIF"))):
    prefix_map.setdefault(os.path.basename(fp).split("_")[0], fp)

print(f"{len(prefix_map)} prefixes from {args.input_dir}")
for prefix, fp in prefix_map.items():
    img = preprocess_experimental_backgroundblack(fp)
    outs = pipeline.process(img, **ARGS)
    for i, out in enumerate(outs, start=1):
        cv2.imwrite(os.path.join(args.out_dir, f"{prefix}_{i}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"Done: {prefix}")
