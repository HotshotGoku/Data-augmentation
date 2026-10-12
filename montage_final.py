"""Final escalation montage: source | real | in-domain SD1.5 | pilot Flux | scaled-fp8 | bf16-CN6.
The money figure: SD1.5 wins realism; Flux improves with every lever (CMMD 8.78->4.84) but plateaus
above SD1.5. Same 5 evenly-spaced test conditions (non-cherry-picked)."""
import os, glob
from PIL import Image, ImageDraw, ImageFont
BASE = "/hpc/group/youlab/sa603/data/branching_256"
R = "/hpc/group/youlab/sa603/code/Data_augmentation"
COLS = [("source", f"{BASE}/test/sources/%s_src.TIF", (220,220,220)),
        ("real", f"{BASE}/test/reals/%s_Rep1.TIF", (220,220,220)),
        ("SD1.5  0.32/3.4", f"{R}/sd15_branching_eval/%s_1.png", (134,239,172)),
        ("pilot  0.52/8.8", f"{R}/flux_cn_runs/combo2/cn065pg06/%s_1.png", (252,165,165)),
        ("scaled 0.50/5.5", f"{R}/flux_cn_runs/scaled/eval_cn10/%s_1.png", (251,191,36)),
        ("bf16   0.50/4.8", f"{R}/flux_cn_runs/scaled_bf16/eval_cn10/%s_1.png", (125,211,252))]
srcs = sorted(glob.glob(f"{BASE}/test/sources/*_src.TIF"))
conds = [os.path.basename(s).split("_")[0] for s in srcs]
NC = int(os.environ.get("NCOND", "5"))
CONDS = [conds[int(i*(len(conds)-1)/(NC-1))] for i in range(NC)]
TILE, GAP, LAB, HEAD = int(os.environ.get("TILE", "200")), 6, 90, 24
f = ImageFont.load_default()
def load(p):
    try: return Image.open(p).convert("RGB").resize((TILE, TILE))
    except Exception: return Image.new("RGB", (TILE, TILE), (40, 40, 40))
W = LAB + len(COLS)*(TILE+GAP); H = HEAD + len(CONDS)*(TILE+GAP+HEAD)
im = Image.new("RGB", (W, H), (10, 10, 12)); d = ImageDraw.Draw(im)
d.text((4, 8), "div/CMMD", fill=(150,150,150), font=f)
for ci, (name, _, col) in enumerate(COLS):
    d.text((LAB+ci*(TILE+GAP), 8), name, fill=col, font=f)
y = HEAD
for cond in CONDS:
    d.text((4, y+TILE//2), cond, fill=(130,200,250), font=f)
    for ci, (_, pat, _) in enumerate(COLS):
        x = LAB+ci*(TILE+GAP); p = pat % cond
        if os.path.exists(p): im.paste(load(p), (x, y))
        else: d.rectangle([x, y, x+TILE, y+TILE], outline=(80,80,80))
    y += TILE+GAP+HEAD
out = os.environ.get("OUTNAME", f"{R}/flux_escalation_final.png"); im.save(out); print("saved", out, im.size)
