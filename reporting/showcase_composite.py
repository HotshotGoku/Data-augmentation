"""Showcase grid: generated colonies standardized into consistent full-rimmed petri dishes."""
import glob, os, numpy as np
from PIL import Image, ImageDraw, ImageFont
from dish_composite import composite, load
GEN = "/hpc/group/youlab/sa603/code/Data_augmentation/dropout_realsweep/drop_s1.0_g7.0"
files = sorted(glob.glob(f"{GEN}/*.png"))
# one sample per condition, spread across conditions
byc = {}
for f in files:
    c = os.path.basename(f).split("_")[0]; byc.setdefault(c, f)
conds = sorted(byc, key=lambda c: int("".join(ch for ch in c if ch.isdigit()) or 0))
NC, NR = 7, 4; sel = [conds[int(i*(len(conds)-1)/(NC*NR-1))] for i in range(NC*NR)]
TILE, GAP, TOP = 180, 6, 44
im = Image.new("RGB", (16 + NC*(TILE+GAP), TOP + NR*(TILE+GAP) + 10), (11,12,15)); d = ImageDraw.Draw(im)
try: fB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
except Exception: fB = ImageFont.load_default()
d.text((16, 12), "Generated branching colonies — standardized petri dishes (consistent size · full rim · colony inside)", font=fB, fill=(134,239,172))
for i, c in enumerate(sel):
    r, cc = divmod(i, NC)
    img = composite(load(byc[c]))
    im.paste(Image.fromarray(img).resize((TILE,TILE)), (16 + cc*(TILE+GAP), TOP + r*(TILE+GAP)))
out = "/hpc/group/youlab/sa603/code/Data_augmentation/composite_showcase.png"; im.save(out); print("saved", out, im.size)
