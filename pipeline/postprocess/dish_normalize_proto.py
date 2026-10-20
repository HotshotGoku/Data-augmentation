"""Prototype: detect the petri dish, quantify its size variation, then normalize (scale dish to a
fixed radius, center it, mask outside the circle). Builds a before/after montage + reports the variation."""
import cv2, numpy as np, glob, os
from PIL import Image, ImageDraw, ImageFont
D = "/hpc/group/youlab/sa603/data/branching_256"

def detect_dish(img):
    """Return (cx, cy, r) of the petri dish. Hough on the rim; fallback = non-black extent."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gb = cv2.medianBlur(g, 5)
    circles = cv2.HoughCircles(gb, cv2.HOUGH_GRADIENT, dp=1.2, minDist=256,
                               param1=60, param2=30, minRadius=85, maxRadius=130)
    if circles is not None:
        x, y, r = circles[0][0]; return float(x), float(y), float(r)
    _, th = cv2.threshold(g, 12, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (x, y), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea)); return x, y, r

def normalize(img, target_r=120, size=256):
    x, y, r = detect_dish(img)
    s = target_r / r
    M = cv2.getRotationMatrix2D((x, y), 0, s); M[0, 2] += size / 2 - x; M[1, 2] += size / 2 - y
    out = cv2.warpAffine(img, M, (size, size))
    mask = np.zeros((size, size), np.uint8); cv2.circle(mask, (size // 2, size // 2), target_r, 255, -1)
    out[mask == 0] = 0
    return out

files = sorted(glob.glob(f"{D}/canon/*_Rep1.TIF"))
# quantify dish-radius variation across many reals
radii = []
for f in files[:120]:
    im = np.array(Image.open(f).convert("RGB").resize((256, 256)))
    try: radii.append(detect_dish(im)[2])
    except Exception: pass
radii = np.array(radii)
print(f"DISH RADIUS across {len(radii)} reals: mean {radii.mean():.1f}px  std {radii.std():.1f}  "
      f"min {radii.min():.1f}  max {radii.max():.1f}  (range {radii.max()-radii.min():.1f}px = {100*(radii.max()-radii.min())/radii.mean():.0f}% of mean)", flush=True)

# before/after montage on 5 samples
SEL = [files[int(i*(len(files)-1)/4)] for i in range(5)]
TILE = 240
im = Image.new("RGB", (40 + 2*(TILE+8), 40 + len(SEL)*(TILE+8)), (11,12,15)); d = ImageDraw.Draw(im)
try: fB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except Exception: fB = ImageFont.load_default()
d.text((40, 10), "original (dish size/position varies)", font=fB, fill=(252,165,165))
d.text((40+TILE+8, 10), "normalized (dish fixed + circular mask)", font=fB, fill=(134,239,172))
for ri, f in enumerate(SEL):
    orig = np.array(Image.open(f).convert("RGB").resize((256, 256)))
    norm = normalize(orig)
    im.paste(Image.fromarray(orig).resize((TILE,TILE)), (40, 40+ri*(TILE+8)))
    im.paste(Image.fromarray(norm).resize((TILE,TILE)), (40+TILE+8, 40+ri*(TILE+8)))
out = f"/hpc/group/youlab/sa603/code/Data_augmentation/dish_normalize_proto.png"
im.save(out); print("saved", out, im.size)
