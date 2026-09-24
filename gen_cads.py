"""Generate replicates with CADS (condition-annealed sampling) on the current ControlNet model.

NEW FILE - touches no existing code. Loads the model through `pipeline` (FT_EVAL_CKPT), then
monkey-patches pipeline.ddim_sampler with a CADSSampler built from CADS_* env vars. Same I/O
contract as reptorep_infer_rattray.py: writes {prefix}_{i}.png from *.TIF sources, so the
existing eval_metrics.py scores it unchanged.

CADS OFF (== stock sampler): set CADS_TAU2 <= CADS_TAU1 (e.g. CADS_TAU2=0). Then only INFER_ETA
differs from the stock path, which lets this same script run the plain eta>0 comparison arm.

  FT_EVAL_CKPT=<ckpt> CADS_S=0.1 CADS_TAU1=0.6 CADS_TAU2=0.9 INFER_ETA=0.0 \
    python gen_cads.py --input_dir <sources> --out_dir <gen>
"""
import os, glob, cv2, argparse
import pipeline  # loads FT_EVAL_CKPT on the GPU at import; provides .model, .process, .ddim_sampler
import Data_augmentation.utils.shared_resources_config as _srcfg  # noqa: F401
from ddim_cads import CADSSampler

ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", required=True)
ap.add_argument("--out_dir", required=True)
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# The ONLY change vs the stock infer path: swap in the CADS sampler (env-driven).
# process() reads the module global pipeline.ddim_sampler at call time, so this takes effect.
pipeline.ddim_sampler = CADSSampler(
    pipeline.model,
    tau1=float(os.environ.get("CADS_TAU1", "0.6")),
    tau2=float(os.environ.get("CADS_TAU2", "0.9")),
    s=float(os.environ.get("CADS_S", "0.1")),
    psi=float(os.environ.get("CADS_PSI", "1.0")),
    rescale=os.environ.get("CADS_RESCALE", "1") == "1",
    cads_uncond=os.environ.get("CADS_UNCOND", "1") == "1",
)
print(f"[gen_cads] CADS s={os.environ.get('CADS_S','0.1')} tau1={os.environ.get('CADS_TAU1','0.6')} "
      f"tau2={os.environ.get('CADS_TAU2','0.9')} rescale={os.environ.get('CADS_RESCALE','1')} "
      f"eta={os.environ.get('INFER_ETA','0.0')}", flush=True)

ARGS = {"prompt": "", "a_prompt": "",
        "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
        "num_samples": 2, "image_resolution": 256,
        "ddim_steps": int(os.environ.get("INFER_DDIM", "50")),
        "guess_mode": os.environ.get("INFER_GUESS", "0") == "1",
        "strength": float(os.environ.get("INFER_STRENGTH", "1.0")),
        "scale": float(os.environ.get("INFER_SCALE", "9.0")),
        "seed": 729397049, "eta": float(os.environ.get("INFER_ETA", "0.0"))}

prefix_map = {}
for fp in sorted(glob.glob(os.path.join(args.input_dir, "*.TIF"))):
    prefix_map.setdefault(os.path.basename(fp).split("_")[0], fp)

print(f"{len(prefix_map)} test sources from {args.input_dir}", flush=True)
for prefix, fp in prefix_map.items():
    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  # already canonical 256px
    outs = pipeline.process(img, **ARGS)
    for i, out in enumerate(outs, start=1):
        cv2.imwrite(os.path.join(args.out_dir, f"{prefix}_{i}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("[gen_cads] done", flush=True)
