"""Branching bake-off comparison montage: does high CMMD (~50-56) mean bad images?
Rows = conditions; cols = Source | Real#1 | Real#2 | PixCell-256 warm | PixCell-256 fb+LoRA | PixCell-1024 fb+LoRA.
"""
import os, glob
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
GEN = "staging/hpc/group/youlab/sa603/code/Data_augmentation"
ARMS = {
    "warm256": f"{GEN}/pixcell_gen/branching256_ep30_ddpm",
    "fb256":   f"{GEN}/pixcell1024_gen/branching256fb_ddpm",
    "fb1024":  f"{GEN}/pixcell1024_gen/branching_ddpm",
}
CONDS = ["ffc0", "ffc112", "ffc147"]
TILE, GAP, LG = 300, 12, 100
TITLE_H, HEAD_H, CAP_H = 104, 92, 92

def font(sz, bold=False):
    for p in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

def load(path, size=TILE):
    im = Image.open(path).convert("RGB")
    return im.resize((size, size), Image.LANCZOS)

COLS = [
    ("Source", "(sim input)", None),
    ("Real replicate #1", "", None),
    ("Real replicate #2", "", None),
    ("PixCell-256 warm-start", "CMMD 49.8 · real 0.650 · div 0.254", "warm256"),
    ("PixCell-256 from-base+LoRA", "CMMD 56.3 · real 0.776 · div 0.424", "fb256"),
    ("PixCell-1024 from-base+LoRA", "CMMD 55.1 · real 0.599 · div 0.342", "fb1024"),
]
NC = len(COLS)
W = LG + NC * TILE + (NC - 1) * GAP + 16
H = TITLE_H + HEAD_H + len(CONDS) * (TILE + GAP) + CAP_H
canvas = Image.new("RGB", (W, H), (14, 15, 19))
d = ImageDraw.Draw(canvas)

fT, fH, fh, fR, fC = font(23, True), font(15, True), font(12), font(13), font(15, True)
TITLE, GRAY, GREEN, CORAL, BLUE = (233,234,236), (154,160,166), (134,239,172), (252,165,165), (125,211,252)

d.text((LG-8, 14), "Branching domain — do high CMMD scores (~50–56) mean the images are bad? Judge visually.", font=fT, fill=TITLE)
d.text((LG-8, 46), "CMMD = distribution-level CLIP distance (lower=better). SD1.5 generalist reference on this domain: CMMD 8.8.  Real-vs-real realism floor ~ 0.32 (1.0 = as close to real as two reals are).", font=fh, fill=GRAY)
d.text((LG-8, 66), "Cols 2–3 are two REAL replicates of the same colony = the natural-variation yardstick. Arms 4–6 = one generated sample each.", font=fh, fill=BLUE)

x0 = LG
for ci, (t1, t2, _) in enumerate(COLS):
    cx = x0 + ci * (TILE + GAP)
    col = TITLE if ci < 3 else (GREEN if ci in (3,5) else CORAL)
    d.text((cx, TITLE_H + 8), t1, font=fH, fill=col)
    if t2:
        d.text((cx, TITLE_H + 30), t2, font=fh, fill=GRAY)

y0 = TITLE_H + HEAD_H
for ri, cond in enumerate(CONDS):
    ry = y0 + ri * (TILE + GAP)
    d.text((6, ry + TILE//2 - 8), cond, font=fR, fill=BLUE)
    reals = sorted(glob.glob(f"{BASE}/reals/{cond}_Rep*.TIF"))
    tiles = [
        f"{BASE}/sources/{cond}_src.TIF",
        reals[0] if reals else None,
        reals[1] if len(reals) > 1 else None,
        f"{BASE}/{ARMS['warm256']}/{cond}_1.png",
        f"{BASE}/{ARMS['fb256']}/{cond}_1.png",
        f"{BASE}/{ARMS['fb1024']}/{cond}_1.png",
    ]
    for ci, tp in enumerate(tiles):
        cx = x0 + ci * (TILE + GAP)
        if tp and os.path.exists(tp):
            canvas.paste(load(tp), (cx, ry))
        else:
            d.rectangle([cx, ry, cx+TILE, ry+TILE], outline=(60,60,66))
            d.text((cx+8, ry+8), "missing", font=fh, fill=CORAL)

cy = y0 + len(CONDS) * (TILE + GAP) + 6
d.text((LG-8, cy), "Takeaway: on branching the PixCell arms are visibly weak — washed-out, low-contrast, grayer than real (pixel std ~15–22 vs 27 for real; the branch structure fades out).", font=fh, fill=CORAL)
d.text((LG-8, cy+20), "So high CMMD (~50–56) is corroborated by eye AND by realism (0.60–0.78 vs a 0.32 real-vs-real floor; SD1.5 gets realism 0.559 / CMMD 8.8). PixCell is the wrong backbone for dense branching.", font=fh, fill=GRAY)
d.text((LG-8, cy+40), "Nuance: higher-res fb1024 scores BEST on per-sample realism (0.599) yet looks MOST washed — LPIPS rewards its 1024px texture while global contrast is off. On cell-like multiplexed the verdict flips (PixCell CMMD 5.8 < SD1.5 12.6).", font=fh, fill=GRAY)

out = f"{BASE}/branching_bakeoff_comparison.png"
canvas.save(out)
print("saved", out, canvas.size)
