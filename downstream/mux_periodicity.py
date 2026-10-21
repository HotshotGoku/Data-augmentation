"""Mux pattern PERIODICITY labeler — a NEW continuous morphological downstream target for the
multiplexed dataset, complementary to the aTc/IPTG decode task.

The multiplexed patterns are RADIAL PETALS (angular lobes), not rings or stripes: aTc (row) drives
red expression + the number/strength of radial spikes, IPTG (col) drives green. So periodicity =
ANGULAR lobe count. We take the colony's radius-vs-angle profile r(theta), FFT it, and the dominant
angular frequency k in 2..24 is the lobe count; its spectral share is the lobe strength.

Rotation-invariant (rotation shifts theta but not the lobe COUNT/strength), so a generative augmenter
can add label-carrying variation that rotation cannot -- the regime where aug can beat rotation.

Outputs per image: n_lobes, lobe_power (peak share of the angular spectrum), perim_roughness (std/mean of r).

Usage: python mux_periodicity.py --images "/path/*.png" --out mux_periodicity.csv
"""
import os, glob, argparse, csv
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage import measure, morphology


def colony_mask(rgb):
    """Bright green+red colony on black; Otsu on (R+G), keep the largest component (drops dish ring/specks)."""
    rg = rgb[..., 0].astype(float) + rgb[..., 1]
    rg /= (rg.max() + 1e-9)
    m = morphology.remove_small_objects(rg > max(threshold_otsu(rg), 0.12), 64)
    lab = measure.label(m)
    if lab.max() == 0:
        return m, rg
    return lab == max(measure.regionprops(lab), key=lambda r: r.area).label, rg


def angular_profile(mask, nbin=360):
    """Max colony radius per angular bin, centered on the colony centroid; empty bins filled circularly."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    cy, cx = ys.mean(), xs.mean()
    ang = (np.arctan2(ys - cy, xs - cx) + 2 * np.pi) % (2 * np.pi)
    rad = np.hypot(ys - cy, xs - cx)
    bins = np.minimum((ang / (2 * np.pi) * nbin).astype(int), nbin - 1)
    r = np.full(nbin, np.nan)
    for b in np.unique(bins):
        r[b] = rad[bins == b].max()
    good = ~np.isnan(r)
    if good.sum() < nbin // 2:
        return None
    return np.interp(np.arange(nbin), np.arange(nbin)[good], r[good], period=nbin)


def periodicity(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    r = angular_profile(colony_mask(rgb)[0])
    if r is None:
        return None
    spec = np.abs(np.fft.rfft(r - r.mean())) ** 2
    ks = np.arange(len(spec))
    band = (ks >= 2) & (ks <= 24)
    name, rough = os.path.basename(path), float(r.std() / (r.mean() + 1e-9))
    if not band.any() or spec[band].sum() == 0:
        return dict(name=name, n_lobes=0, lobe_power=0.0, perim_roughness=rough)
    k = int(ks[band][np.argmax(spec[band])])
    return dict(name=name, n_lobes=k, lobe_power=float(spec[k] / (spec[1:].sum() + 1e-9)),
                perim_roughness=rough)


def _selfcheck():
    N = 256; yy, xx = np.mgrid[0:N, 0:N]; c = N / 2
    ang = np.arctan2(yy - c, xx - c); rad = np.hypot(yy - c, xx - c)
    flower = rad < 80 * (1 + 0.3 * np.cos(8 * ang))     # 8-petal flower
    r = angular_profile(flower); spec = np.abs(np.fft.rfft(r - r.mean())) ** 2
    ks = np.arange(len(spec)); k = int(ks[(ks >= 2) & (ks <= 24)][np.argmax(spec[(ks >= 2) & (ks <= 24)])])
    assert k == 8, f"flower n_lobes={k} expected 8"
    rc = angular_profile(rad < 80)                      # plain circle
    assert rc.std() / rc.mean() < 0.05, "circle should be smooth"
    print(f"selfcheck OK (8-petal flower -> n_lobes={k}, circle roughness={rc.std()/rc.mean():.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="")
    ap.add_argument("--out", default="mux_periodicity.csv")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); raise SystemExit
    rows = [f for f in (periodicity(p) for p in sorted(glob.glob(a.images))) if f]
    keys = ["name", "n_lobes", "lobe_power", "perim_roughness"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in keys})
    print(f"wrote {len(rows)} rows -> {a.out}")
