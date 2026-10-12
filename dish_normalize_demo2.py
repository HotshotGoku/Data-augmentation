"""Honest demo: measure dish radius in REAL vs GENERATED images (the mismatch the user noticed), and
show normalization applied to GENERATED images. Columns: REAL (ref) | GEN original | GEN normalized."""
import cv2, numpy as np, glob, os
from PIL import Image, ImageDraw, ImageFont
D = "/hpc/group/youlab/sa603/data/branching_256"
GEN = "/hpc/group/youlab/sa603/code/Data_augmentation/dropout_realsweep/drop_s1.0_g7.0"  # realistic generated

from dish_normalize import detect_dish, normalize   # uses the fixed version (mask_r > target_r -> rim preserved)
def rad(files):
    rs = []
    for f in files[:110]:
        try: rs.append(detect_dish(np.array(Image.open(f).convert("RGB").resize((256,256))))[2])
        except Exception: pass
    return np.array(rs)

rr = rad(sorted(glob.glob(f"{D}/test/reals/*.TIF")))
gr = rad(sorted(glob.glob(f"{GEN}/*.png")))
print(f"REAL dish radius: mean {rr.mean():.1f} std {rr.std():.1f} range[{rr.min():.0f},{rr.max():.0f}]")
print(f"GEN  dish radius: mean {gr.mean():.1f} std {gr.std():.1f} range[{gr.min():.0f},{gr.max():.0f}]")
print(f"REAL-vs-GEN mean dish-size mismatch: {abs(rr.mean()-gr.mean()):.1f}px")

conds = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{GEN}/*.png")} &
               {os.path.basename(f).split("_")[0] for f in glob.glob(f"{D}/test/reals/*.TIF")})
SEL = [conds[int(i*(len(conds)-1)/3)] for i in range(4)]
TILE = 240
cols = ["REAL (reference)", "GENERATED (original)", "GENERATED (normalized)"]
colc = [(134,239,172),(252,165,165),(125,211,252)]
im = Image.new("RGB", (40+3*(TILE+8), 44+len(SEL)*(TILE+8)), (11,12,15)); d = ImageDraw.Draw(im)
try: fB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except Exception: fB = ImageFont.load_default()
for ci,(c,cc) in enumerate(zip(cols,colc)): d.text((40+ci*(TILE+8), 12), c, font=fB, fill=cc)
for ri,cd in enumerate(SEL):
    real = np.array(Image.open(f"{D}/test/reals/{cd}_Rep1.TIF").convert("RGB").resize((256,256)))
    g = sorted(glob.glob(f"{GEN}/{cd}_*.png"))[0]; geno = np.array(Image.open(g).convert("RGB").resize((256,256)))
    genn = normalize(geno)
    for ci,arr in enumerate([real, geno, genn]):
        im.paste(Image.fromarray(arr).resize((TILE,TILE)), (40+ci*(TILE+8), 44+ri*(TILE+8)))
out = "/hpc/group/youlab/sa603/code/Data_augmentation/dish_normalize_demo2.png"; im.save(out); print("saved", out, im.size, SEL)
