"""Scope multiplexed generalization: do multiplexed images have petri dishes (for standardization)?
Montage a few reals + measure dish detectability/variation."""
import glob, numpy as np, os
from PIL import Image, ImageDraw, ImageFont
from dish_normalize import detect_dish
D = "/hpc/group/youlab/sa603/data/multiplexed_eval"
reals = sorted(glob.glob(f"{D}/reals_heldout/*.TIF"))
# dish detection stats
rs = []
for f in reals[:80]:
    try: rs.append(detect_dish(np.array(Image.open(f).convert("RGB").resize((256,256))))[2])
    except Exception: pass
rs = np.array(rs)
print(f"MULTIPLEXED dish radius: mean {rs.mean():.1f} std {rs.std():.1f} range[{rs.min():.0f},{rs.max():.0f}] (n={len(rs)})")
print(f"(branching was: real mean 117.9 std 3.8; if mux std is large/erratic, dishes may not be cleanly present)")
# montage of 8 multiplexed reals
SEL = [reals[int(i*(len(reals)-1)/7)] for i in range(8)]
TILE = 220
im = Image.new("RGB", (16+4*(TILE+6), 40+2*(TILE+6)), (11,12,15)); d = ImageDraw.Draw(im)
try: fB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except Exception: fB = ImageFont.load_default()
d.text((16, 12), "Multiplexed (cell-like) real images — do these have petri dishes like branching?", font=fB, fill=(235,235,235))
for i, f in enumerate(SEL):
    r, c = divmod(i, 4)
    im.paste(Image.fromarray(np.array(Image.open(f).convert("RGB").resize((TILE,TILE)))), (16+c*(TILE+6), 40+r*(TILE+6)))
out = "/hpc/group/youlab/sa603/code/Data_augmentation/mux_scope.png"; im.save(out); print("saved", out)
