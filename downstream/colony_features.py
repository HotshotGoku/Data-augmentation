"""Colony-feature auto-labeler for the downstream feature-regression task.
Segments a colony from a dish image and computes structure/orientation-sensitive features that
plain rotation augmentation can't enrich: shape factor, fractal dimension, lacunarity, edge complexity.
No Fiji dependency (pure Python: skimage + scipy). Reuses the You-lab 2021 shape-factor definition
(log10 circularity) and Rattray-2023-style Sobel edge-complexity stats.

Usage: python colony_features.py --images "model_results/refs_tif/*.TIF" --out features.csv [--montage overlay.png]
"""
import os, glob, argparse, csv
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, binary_fill_holes
from scipy.stats import skew, kurtosis
from skimage import filters, measure, morphology, segmentation


def load_gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float64)


def _dish_roi(g):
    """Stage 1: the agar dish = largest bright blob vs dark surround; eroded to drop the bright rim."""
    thr = filters.threshold_otsu(g)
    dish = binary_fill_holes(morphology.remove_small_objects(g > thr, min_size=g.size // 50))
    lab = measure.label(dish)
    if lab.max() == 0:
        return None
    biggest = max(measure.regionprops(lab), key=lambda r: r.area)
    dish = lab == biggest.label
    margin = max(3, min(g.shape) // 55)
    return morphology.binary_erosion(dish, morphology.disk(margin))


def segment_colony(gray):
    """Two-stage: find the dish ROI, then threshold the (brighter) colony structure inside it.
    Keeps the ramified boundary (light closing only) so branch structure survives."""
    g = filters.gaussian(gray, sigma=1.0, preserve_range=True)
    dish = _dish_roi(g)
    if dish is None or dish.sum() < 200:
        return None
    vals = g[dish]
    thr1 = filters.threshold_otsu(vals)                 # colony (bright) vs agar (mid) inside dish
    colony = (g > thr1) & dish
    colony = morphology.binary_closing(colony, morphology.disk(2))  # reconnect branch arms, keep shape
    colony = morphology.remove_small_objects(colony, min_size=max(32, g.size // 6000))
    if colony.sum() < dish.sum() * 0.002:               # ~empty dish / no colony
        return None
    lab = measure.label(colony)
    return lab == max(measure.regionprops(lab), key=lambda r: r.area).label


def fractal_dimension(binary):
    """Box-counting dimension of the boundary (ramification). D=slope of log(count) vs log(1/size)."""
    edges = binary ^ morphology.binary_erosion(binary)
    pts = edges
    n = min(pts.shape)
    sizes = 2 ** np.arange(1, int(np.log2(n)) )
    counts = []
    for s in sizes:
        # number of s x s boxes containing any boundary pixel
        sh = (pts.shape[0] // s, pts.shape[1] // s)
        if sh[0] == 0 or sh[1] == 0:
            counts.append(1); continue
        trimmed = pts[:sh[0]*s, :sh[1]*s]
        blocks = trimmed.reshape(sh[0], s, sh[1], s).any(axis=(1, 3))
        counts.append(max(1, int(blocks.sum())))
    counts = np.array(counts, float)
    coeffs = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
    return float(coeffs[0])


def lacunarity(binary, box=None):
    """Gliding-box lacunarity Lambda(r)=E[m^2]/E[m]^2 (gappiness/texture heterogeneity)."""
    if box is None:
        box = max(4, min(binary.shape) // 16)
    m = binary.astype(np.float64)
    s = uniform_filter(m, size=box) * (box * box)  # box mass at each position
    mean = s.mean()
    if mean == 0:
        return 0.0
    return float((s.var() + mean**2) / (mean**2))


def edge_complexity(gray, mask):
    """Sobel gradient-magnitude stats inside the colony (Rattray-2023-style complexity)."""
    gmag = filters.sobel(gray / (gray.max() + 1e-9))
    v = gmag[mask]
    if v.size < 10:
        return dict(edge_std=0.0, edge_skew=0.0, edge_kurt=0.0)
    return dict(edge_std=float(v.std()), edge_skew=float(skew(v)), edge_kurt=float(kurtosis(v)))


def features(path):
    gray = load_gray(path)
    mask = segment_colony(gray)
    if mask is None:
        return None, None
    r = measure.regionprops(mask.astype(int))[0]
    area, perim = float(r.area), float(r.perimeter or 1.0)
    circ = min(1.0, 4 * np.pi * area / (perim ** 2))
    f = dict(
        name=os.path.basename(path),
        area_frac=area / gray.size,
        circularity=circ,
        shape_factor=float(np.log10(max(circ, 1e-6))),  # You-lab 2021: round~0, branched<<0
        solidity=float(r.solidity),
        eccentricity=float(r.eccentricity),
        fractal_dim=fractal_dimension(mask),
        lacunarity=lacunarity(mask),
    )
    f.update(edge_complexity(gray, mask))
    return f, mask


def _selfcheck():
    sq = np.zeros((256, 256), bool); sq[64:192, 64:192] = True
    d_sq = fractal_dimension(sq)                 # smooth square boundary -> ~1 (1D contour)
    assert 0.9 <= d_sq < 1.4, f"square boundary D={d_sq:.2f} expected ~1"
    # a ramified 'plus' with thin arms has a rougher/longer boundary than a compact square
    cross = np.zeros((256, 256), bool); cross[120:136, 20:236] = True; cross[20:236, 120:136] = True
    assert fractal_dimension(cross) >= d_sq - 0.05, "branched shape should not be smoother than a square"
    assert lacunarity(sq) > 1.0
    print(f"selfcheck OK (square boundary D={d_sq:.2f}, cross D={fractal_dimension(cross):.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="model_results/refs_tif/*.TIF")
    ap.add_argument("--out", default="colony_features.csv")
    ap.add_argument("--montage", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); raise SystemExit

    paths = sorted(glob.glob(a.images))
    rows, overlays = [], []
    for p in paths:
        f, mask = features(p)
        if f is None:
            print(f"  SKIP (no colony): {os.path.basename(p)}"); continue
        rows.append(f)
        if a.montage:
            g = load_gray(p); rgb = np.stack([g]*3, -1); rgb = (rgb/ (rgb.max()+1e-9) *255).astype(np.uint8)
            b = mask ^ morphology.binary_erosion(mask)
            rgb[b] = [255, 60, 60]
            overlays.append((os.path.basename(p), Image.fromarray(rgb).resize((256, 256))))

    keys = ["name","area_frac","circularity","shape_factor","solidity","eccentricity","fractal_dim","lacunarity","edge_std","edge_skew","edge_kurt"]
    with open(a.out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys); wtr.writeheader()
        for r in rows: wtr.writerow({k: (round(r[k],4) if isinstance(r[k],float) else r[k]) for k in keys})
    print(f"wrote {len(rows)} rows -> {a.out}")

    if a.montage and overlays:
        cols = 6; rows_n = (len(overlays)+cols-1)//cols; T=256
        canvas = Image.new("RGB",(cols*T, rows_n*(T+18)),(14,15,19))
        from PIL import ImageDraw
        d = ImageDraw.Draw(canvas)
        for i,(nm,im) in enumerate(overlays):
            x,y=(i%cols)*T,(i//cols)*(T+18); canvas.paste(im,(x,y)); d.text((x+3,y+T+2), nm, fill=(200,200,200))
        canvas.save(a.montage); print("montage ->", a.montage)
