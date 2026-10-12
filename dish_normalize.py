"""Dish normalization: detect the petri dish, scale it to a fixed radius, center it, mask outside the
circle. Removes the ~17% dish-size nuisance variation + the model's ~12% dish inflation. CLI: src_dir dst_dir."""
import cv2, numpy as np, os, glob, sys
from PIL import Image
TARGET_R = 118; SIZE = 256    # scale the detected dish rim to exactly the real-dish mean radius (117.9)
MASK_R = 128                  # mask only OUTSIDE the inscribed circle (= the 4 corners, black in reals too) -> full glass rim preserved at real scale

def detect_dish(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); gb = cv2.medianBlur(g, 5)
    c = cv2.HoughCircles(gb, cv2.HOUGH_GRADIENT, dp=1.2, minDist=SIZE, param1=60, param2=30, minRadius=80, maxRadius=135)
    if c is not None:
        x, y, r = c[0][0]; return float(x), float(y), float(r)
    _, th = cv2.threshold(g, 12, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        (x, y), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea)); return x, y, r
    return SIZE / 2.0, SIZE / 2.0, float(TARGET_R)

def normalize(img, target_r=TARGET_R, mask_r=MASK_R, size=SIZE):
    x, y, r = detect_dish(img)
    if not (70 < r < 140):                      # detection sanity -> default to centered mean dish
        x, y, r = size / 2.0, size / 2.0, float(target_r)
    s = target_r / r
    M = cv2.getRotationMatrix2D((x, y), 0, s); M[0, 2] += size / 2 - x; M[1, 2] += size / 2 - y
    out = cv2.warpAffine(img, M, (size, size))
    m = np.zeros((size, size), np.uint8); cv2.circle(m, (size // 2, size // 2), mask_r, 255, -1)
    out[m == 0] = 0                             # mask OUTSIDE mask_r (> target_r) -> glass rim preserved
    return out

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True); n = 0
    for f in sorted(glob.glob(src + "/*.TIF") + glob.glob(src + "/*.png")):
        im = np.array(Image.open(f).convert("RGB").resize((SIZE, SIZE)))
        Image.fromarray(normalize(im)).save(os.path.join(dst, os.path.splitext(os.path.basename(f))[0] + ".png")); n += 1
    print(f"normalized {n} -> {dst}", flush=True)
