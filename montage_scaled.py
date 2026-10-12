"""Progression montage: source | real | in-domain SD1.5 x2 | pilot Flux | scaled Flux x2.
Shows whether scaling closed the gap. Same 5 evenly-spaced test conditions as the verify montage."""
import os, glob
from PIL import Image, ImageDraw, ImageFont
BASE = "/hpc/group/youlab/sa603/data/branching_256"
SD15 = "/hpc/group/youlab/sa603/code/Data_augmentation/sd15_branching_eval"
PILOT = "/hpc/group/youlab/sa603/code/Data_augmentation/flux_cn_runs/combo2/cn065pg06"
SCALED = "/hpc/group/youlab/sa603/code/Data_augmentation/flux_cn_runs/scaled/eval_cn10"
srcs = sorted(glob.glob(f"{BASE}/test/sources/*_src.TIF"))
conds = [os.path.basename(s).split("_")[0] for s in srcs]
CONDS = [conds[int(i*(len(conds)-1)/4)] for i in range(5)]
TILE, GAP, LAB, HEAD = 200, 6, 90, 22
COLS = ["source", "real", "SD1.5 a", "SD1.5 b", "pilot Flux", "scaled a", "scaled b"]
f = ImageFont.load_default()
def load(p):
    try: return Image.open(p).convert("RGB").resize((TILE, TILE))
    except Exception: return Image.new("RGB", (TILE, TILE), (40, 40, 40))
W = LAB + len(COLS)*(TILE+GAP); H = HEAD + len(CONDS)*(TILE+GAP+HEAD)
im = Image.new("RGB", (W, H), (10, 10, 12)); d = ImageDraw.Draw(im)
for ci, c in enumerate(COLS):
    col = (134,239,172) if "SD1.5" in c else (252,165,165) if "pilot" in c else (125,211,252) if "scaled" in c else (220,220,220)
    d.text((LAB+ci*(TILE+GAP), 6), c, fill=col, font=f)
y = HEAD
for cond in CONDS:
    reals = sorted(glob.glob(f"{BASE}/test/reals/{cond}_Rep*.TIF"))
    tiles = [f"{BASE}/test/sources/{cond}_src.TIF", reals[0] if reals else None,
             f"{SD15}/{cond}_1.png", f"{SD15}/{cond}_2.png",
             f"{PILOT}/{cond}_1.png", f"{SCALED}/{cond}_1.png", f"{SCALED}/{cond}_2.png"]
    d.text((4, y+TILE//2), cond, fill=(130,200,250), font=f)
    for ci, t in enumerate(tiles):
        x = LAB+ci*(TILE+GAP)
        if t and os.path.exists(t): im.paste(load(t), (x, y))
        else: d.rectangle([x, y, x+TILE, y+TILE], outline=(80,80,80))
    y += TILE+GAP+HEAD
out = "/hpc/group/youlab/sa603/code/Data_augmentation/scaled_flux_progression.png"
im.save(out); print("saved", out, im.size)
