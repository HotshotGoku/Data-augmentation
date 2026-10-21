"""Petri-dish REMOVAL for preprocessing (Kinshuk feedback). Detect the dish rim, keep only the agar
interior, zero the glass rim + everything outside. Differs from dish_normalize.py, which rescales the
dish to a fixed radius and PRESERVES the rim to kill dish-SIZE nuisance -- here we DROP the rim entirely
so a downstream model can't key on dish glass/size at all (and colony-radius becomes usable).

Reuses detect_dish from dish_normalize.py (Hough rim + contour fallback). BUILD + PREVIEW first; the
decision to apply it to a dataset and re-run generation/scoring is deferred (locked decision #4).

Usage:
  python dish_remove.py --preview "src/*.TIF" --out preview.png   # montage: orig | detected | removed
  python dish_remove.py src_dir dst_dir                           # APPLY: write removed images
  python dish_remove.py --selfcheck
"""
import os, glob, sys, argparse
import cv2, numpy as np
from PIL import Image, ImageDraw
from dish_normalize import detect_dish, SIZE          # same-dir import (script dir is on sys.path)

RIM_FRAC = 0.06            # drop the outer 6% radius (the glass-rim band) so only agar interior survives


def remove_dish(img, rim_frac=RIM_FRAC):
    """img: HxWx3 uint8 RGB. Returns (removed_img, (x, y, r_detected))."""
    x, y, r = detect_dish(img)
    if not (70 < r < 140):                             # detection sanity -> centered mean dish
        x, y, r = img.shape[1] / 2.0, img.shape[0] / 2.0, 118.0
    keep_r = r * (1 - rim_frac)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(m, (int(round(x)), int(round(y))), int(round(keep_r)), 255, -1)
    out = img.copy(); out[m == 0] = 0
    return out, (x, y, r)


def _load(fp):
    return np.array(Image.open(fp).convert("RGB").resize((SIZE, SIZE)))


def _overlay(img, xyr):
    """Draw detected rim (red) + kept-interior boundary (green) for eyeballing the cut."""
    x, y, r = xyr; rgb = np.ascontiguousarray(img.copy())
    cv2.circle(rgb, (int(x), int(y)), int(r), (255, 60, 60), 2)
    cv2.circle(rgb, (int(x), int(y)), int(r * (1 - RIM_FRAC)), (60, 220, 60), 2)
    return rgb


def preview(paths, out, n=8):
    paths = paths[:n]
    T, pad = SIZE, 16
    canvas = Image.new("RGB", (3 * T, len(paths) * (T + pad)), (14, 15, 19))
    d = ImageDraw.Draw(canvas)
    for i, fp in enumerate(paths):
        img = _load(fp); removed, xyr = remove_dish(img)
        y = i * (T + pad)
        for j, im in enumerate([img, _overlay(img, xyr), removed]):
            canvas.paste(Image.fromarray(im), (j * T, y))
        d.text((4, y + 2), f"{os.path.basename(fp)}  r={xyr[2]:.0f}", fill=(230, 230, 230))
    canvas.save(out); print(f"preview ({len(paths)} imgs: orig | detected | removed) -> {out}", flush=True)


def _selfcheck():
    N = SIZE; yy, xx = np.mgrid[0:N, 0:N]; c = N / 2
    img = np.zeros((N, N, 3), np.uint8)
    img[np.hypot(yy - c, xx - c) < 100] = 200          # bright dish r=100 on black
    out, (x, y, r) = remove_dish(img)
    assert 90 < r < 110, f"detected r={r:.1f} expected ~100"
    assert out[0, 0].sum() == 0 and out[-1, -1].sum() == 0, "corners must be removed"
    assert out[int(c), int(c)].sum() > 0, "interior must be kept"
    rb = int(c + r * (1 - RIM_FRAC * 0.5))             # a pixel in the rim band (keep_r < rb < r)
    assert out[int(c), rb].sum() == 0, "rim band must be removed"
    print(f"selfcheck OK (r={r:.1f}, interior kept, rim+corners removed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?"); ap.add_argument("dst", nargs="?")
    ap.add_argument("--preview"); ap.add_argument("--out", default="dish_remove_preview.png")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); sys.exit()
    if a.preview:
        preview(sorted(glob.glob(a.preview)), a.out, a.n); sys.exit()
    if not (a.src and a.dst):
        raise SystemExit("need src_dir dst_dir (apply), or --preview GLOB --out PNG")
    os.makedirs(a.dst, exist_ok=True); n = 0
    for fp in sorted(glob.glob(a.src + "/*.TIF") + glob.glob(a.src + "/*.png")):
        removed, _ = remove_dish(_load(fp))
        Image.fromarray(removed).save(os.path.join(a.dst, os.path.splitext(os.path.basename(fp))[0] + ".png")); n += 1
    print(f"removed dish from {n} imgs -> {a.dst}", flush=True)
