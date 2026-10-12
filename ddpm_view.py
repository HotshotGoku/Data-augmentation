"""See the conditional DDPM output: source | real sibling | DDPM raw | DDPM composited, to diagnose why realism is poor."""
import glob, os, numpy as np
from PIL import Image, ImageDraw, ImageFont
from dish_composite import composite, load
D = "/hpc/group/youlab/sa603/data/branching_256"
G = "/hpc/group/youlab/sa603/code/Data_augmentation/cond_ddpm_runs/gen"
conds = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{G}/*.png")}
               & {os.path.basename(f).split("_")[0] for f in glob.glob(f"{D}/test/reals/*.TIF")})
SEL = [conds[int(i*(len(conds)-1)/3)] for i in range(4)]
TILE = 220; cols = ["source", "real sibling", "DDPM (raw)", "DDPM (composited)"]
colc = [(235,235,235),(134,239,172),(252,165,165),(125,211,252)]
try: fB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except Exception: fB = ImageFont.load_default()
im = Image.new("RGB", (40+4*(TILE+8), 44+len(SEL)*(TILE+8)), (11,12,15)); d = ImageDraw.Draw(im)
d.text((40, 12), "Conditional DDPM output (realism 0.538 vs SD1.5 0.324) — diagnose", font=fB, fill=(235,235,235))
for ci,(c,cc) in enumerate(zip(cols,colc)): d.text((40+ci*(TILE+8), 30), c, font=fB, fill=cc)
for ri,cd in enumerate(SEL):
    g = sorted(glob.glob(f"{G}/{cd}_*.png"))[0]; graw = load(g)
    tiles = [load(f"{D}/test/sources/{cd}_src.TIF"), load(f"{D}/test/reals/{cd}_Rep1.TIF"), graw, composite(graw)]
    for ci,arr in enumerate(tiles): im.paste(Image.fromarray(arr).resize((TILE,TILE)), (40+ci*(TILE+8), 50+ri*(TILE+8)))
out = "/hpc/group/youlab/sa603/code/Data_augmentation/ddpm_view.png"; im.save(out); print("saved", out, SEL)
