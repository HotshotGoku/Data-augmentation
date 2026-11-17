"""Scale-out AUGMENTATION preview: for each dataset, load a few BAKED _pp images and apply the REAL
runtime augment_pair (same fill/brightness the launcher sets) -> tile [baked | aug | aug | aug] so the
user can eyeball what the generator actually trains on. Separate from make_review_montages.py (which
compares RAW->preprocess for preprocessing sign-off); this is for AUG sign-off.

Run on dcc-agent (env pytorch_PA_patternprediction). Writes one PNG per dataset to OUT_DIR.
"""
import os, glob, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pipeline"))
from gen_augment import augment_pair  # noqa

D = "/hpc/group/youlab/sa603/data"
# name -> (glob of baked images, AUG_FILL, AUG_BRIGHTNESS)  -- mirrors each launcher's env
DATASETS = {
    "pakp":     (f"{D}/pakp_256_pp/canon/*.TIF", 255, True),
    "nlev":     (f"{D}/nlev_256_pp/canon/*.TIF", 255, True),
    "selx":     (f"{D}/selx_256_pp/canon/*.TIF", 255, True),
    "rattray":  (f"{D}/external_datasets/rattray_2023/processed_pp/*.png", 255, True),
    "swarmevo": (f"{D}/external_datasets/swarmevo/canon_pp/*/*.png", 0, True),
    "mux":      (f"{D}/multiplexed_eval/reals/*.TIF", 0, False),   # fluor: no bake, rotate/flip only
    "kl":       (f"{D}/ood_kl_2species/*.TIF", 0, False),           # fluor: no bake, rotate/flip only
}
BOOST = os.environ.get("BOOST", "0") == "1"   # show the boosted aug (zoom-out scale + wider photometric)
OUT_DIR = os.environ.get("OUT_DIR", "/hpc/group/youlab/sa603/kinshuk_scaleout_aug" + ("_boost" if BOOST else ""))
N_IMG = 4        # sample images per dataset
N_AUG = 3        # augmented variants per image
os.makedirs(OUT_DIR, exist_ok=True)


def load(fp):
    im = np.array(Image.open(fp).convert("RGB"))
    if im.shape[:2] != (256, 256):
        im = np.array(Image.fromarray(im).resize((256, 256)))
    return im


for name, (pat, fill, bright) in DATASETS.items():
    paths = sorted(glob.glob(pat))[:N_IMG]
    if not paths:
        print(f"[skip] {name}: no images at {pat}", flush=True); continue
    T, pad, H = 256, 10, 20
    cols = 1 + N_AUG
    canvas = Image.new("RGB", (cols * T + pad * (cols - 1), H + len(paths) * (T + pad)), (14, 15, 19))
    d = ImageDraw.Draw(canvas)
    tag = "BOOST " if BOOST else ""
    d.text((4, 4), f"{tag}{name}  fill={fill} brightness={bright}   [baked_pp | aug x{N_AUG}]", fill=(255, 230, 120))
    rng = np.random.default_rng(0)
    for i, fp in enumerate(paths):
        img = load(fp)
        y = H + i * (T + pad)
        canvas.paste(Image.fromarray(img), (0, y))
        for j in range(N_AUG):
            a, _, _ = augment_pair(img, img.copy(), rng, brightness=bright, fill=(fill,) * 3, boost=BOOST)
            canvas.paste(Image.fromarray(a), ((j + 1) * (T + pad), y))
    out = os.path.join(OUT_DIR, f"aug_{name}.png")
    canvas.save(out)
    print(f"[ok] {name}: {len(paths)} imgs -> {out}", flush=True)

print(f"DONE -> {OUT_DIR}", flush=True)
