"""Inference hyperparameter SWEEP for the reptorep augmenter.

Kinshuk asked to (a) do systematic inference, (b) try scale 9.0 -> 15.1, (c) explore other
config knobs. This driver loads the model ONCE (via `import pipeline`, which loads CKPT_PATH_V4 =
our own checkpoint) and generates replicates for every (scale x strength) config across one or
more target folders, writing an eval_config.json next to each config's outputs. Scoring is done
separately by eval_metrics.py in Slurm_scripts/infer_sweep_sa603.sh.

Scientific note: with empty prompts + guess_mode=False the control image is in BOTH the cond and
uncond branches, so `scale` (CFG) mostly affects sharpness, while `strength` (control_scales)
governs how much the output obeys the input shape -> we sweep BOTH.

Usage:
  python reptorep_infer_sweep.py \
      --targets "OOD:/path/to/test_2sp,IND:/path/to/indomain_test:40" \
      --out_root /path/to/infer_sweep_out \
      --scales 9.0,12.0,15.1 --strengths 1.0,1.5,2.0

Upload to: /hpc/group/youlab/sa603/code/Data_augmentation/reptorep_infer_sweep.py
"""
import os
import glob
import json
import itertools
import argparse

import cv2
import pipeline  # loads the trained model (CKPT_PATH_V4, or FT_EVAL_CKPT override) onto the GPU at import
import Data_augmentation.utils.shared_resources_config as shared_resources_config  # noqa: F401
from cldm.preprocess import preprocess_experimental_backgroundblack
from Data_augmentation.utils import tracking

# Fixed inference args; the sweep varies `scale` and `strength`.
BASE = {
    "prompt": "", "a_prompt": "",
    "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
    "num_samples": 2, "image_resolution": 256, "ddim_steps": 50,
    "guess_mode": False, "seed": 729397049, "eta": 0.0,
}


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _parse_targets(s):
    """'TAG:dir[:maxpref],TAG2:dir2' -> [(tag, dir, maxpref_or_None), ...]"""
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        tag, d = parts[0], parts[1]
        mx = int(parts[2]) if len(parts) > 2 and parts[2].strip() else None
        out.append((tag, d, mx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="TAG:dir[:maxpref],TAG2:dir2 ...")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--scales", default="9.0,12.0,15.1")
    ap.add_argument("--strengths", default="1.0,1.5,2.0")
    args = ap.parse_args()

    scales, strengths = _floats(args.scales), _floats(args.strengths)
    targets = _parse_targets(args.targets)

    tracking.write_manifest("infer_sweep", {
        "tag": os.environ.get("SWEEP_TAG", "infersweep"),
        "scales": scales, "strengths": strengths, "targets": args.targets,
        "out_root": args.out_root,
        "ckpt": os.environ.get("FT_EVAL_CKPT", "CKPT_PATH_V4(default sa603 full-data model)"),
        **BASE,
    })

    for tag, d, mx in targets:
        prefix_map = {}
        for fp in sorted(glob.glob(os.path.join(d, "*.TIF"))):
            prefix_map.setdefault(os.path.basename(fp).split("_")[0], fp)
        items = list(prefix_map.items())
        if mx:
            items = items[:mx]
        print(f"[target {tag}] {len(items)} prefixes from {d}", flush=True)

        for scale, strength in itertools.product(scales, strengths):
            cfg = dict(BASE, scale=scale, strength=strength)
            outdir = os.path.join(args.out_root, tag, f"scale{scale}_str{strength}")
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "eval_config.json"), "w") as f:
                json.dump({**cfg, "target": tag, "input_dir": d}, f, indent=2)
            for prefix, fp in items:
                img = preprocess_experimental_backgroundblack(fp)
                outs = pipeline.process(img, **cfg)
                for i, out in enumerate(outs, start=1):
                    cv2.imwrite(os.path.join(outdir, f"{prefix}_{i}.png"),
                                cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            print(f"  [{tag}] scale={scale} strength={strength} -> {outdir}", flush=True)

    print("[infer_sweep] generation done", flush=True)


if __name__ == "__main__":
    main()
