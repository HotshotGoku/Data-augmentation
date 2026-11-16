#!/usr/bin/env python
"""Preprocessing-review montages for Kinshuk's 4-point feedback (2026-11-16).

Builds four before/after montages so Kinshuk can sign off BEFORE we re-run training:
  1_brightfield_preprocess.png  RAW | OURS (dish_remove) | KINSHUK (initialstage_mod)
  2_mux_kl_fluorescence.png     normal images sorted dim->bright (intensity = label, kept as-is)
  3_rattray.png                 RAW (native) | USED (no crop / no inscribe)
  4_swarmevo_myxo.png           RAW (native) | USED (normal, no dish mask)

Run on a DCC CPU compute node (cv2 Hough is compute, not a login-node task):
  conda activate pytorch_PA_patternprediction
  python datasets/preprocess/make_review_montages.py
Outputs to /hpc/group/youlab/sa603/kinshuk_review_v2 (scp'd to the Mac for review).
"""
import os, glob, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

SIZE = 256
OUT = "/hpc/group/youlab/sa603/kinshuk_review_v2"
os.makedirs(OUT, exist_ok=True)

# our petri-dish removal (rim_frac=0.20, black exterior) -- the approach we actually built + used
sys.path.insert(0, "/hpc/group/youlab/sa603/code/Data_augmentation/pipeline/postprocess")
from dish_remove import remove_dish                       # remove_dish(rgb_uint8)->(out,(x,y,r))

KS  = "/hpc/group/youlab/ks723/storage/Exp_images"
EXT = "/hpc/group/youlab/sa603/data/external_datasets"
MPX = "/hpc/group/youlab/sa603/data/multiplexed_eval/reals"


# ---- Kinshuk's recommended function, INLINED VERBATIM from ----
# Simulation_templated_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/cldm/preprocess.py:503
# (inlined, not imported, because his module imports torch at top-level; this fn is cv2+numpy only)
def preprocess_experimental_initialstage_mod(img_path, img_length=256, img_width=256):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=20, maxRadius=600)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x_center, y_center, radius) in circles:
            new_radius = radius - 82
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (x_center, y_center), new_radius, (255, 255, 255), -1)
            img_masked = cv2.bitwise_and(img, img, mask=mask)
            alpha = 1.5; beta = 50
            adjusted_image = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)
            img_masked_rgb = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_masked_rgb, (img_length, img_width))
            mask_resized = cv2.resize(mask, (img_length, img_width), interpolation=cv2.INTER_NEAREST)
            img_masked_resized = cv2.bitwise_and(img_resized, img_resized, mask=mask_resized)
            img_masked_resized[mask_resized == 0] = [255, 255, 255]     # white exterior
            return img_masked_resized
    return None


# ---------- small helpers ----------
def load_stretch(path, size=SIZE):
    """What the training loader does: force to size x size (may stretch)."""
    return np.array(Image.open(path).convert("RGB").resize((size, size)))

def load_fit(path, size=SIZE, bg=(40, 40, 46)):
    """Native aspect, letterboxed -- shows the true shape/edge of the raw image."""
    im = Image.open(path).convert("RGB"); im.thumbnail((size, size))
    c = Image.new("RGB", (size, size), bg)
    c.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.array(c)

def ours_dishremove(path):
    return remove_dish(load_stretch(path))[0]

def even_idx(n, k):
    if n == 0: return []
    if k >= n: return list(range(n))
    if k == 1: return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})

def glob_sorted(d, pat):
    return sorted(glob.glob(os.path.join(d, pat)))

def mean_intensity(path):
    return float(load_stretch(path).mean())

_FONTS = ["/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
def _font(sz):
    for p in _FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()
F_TITLE, F_HEAD, F_CAP = _font(17), _font(13), _font(12)


def grid(rows, out_path, super_title, col_headers=None, row_labels=None):
    """rows: list of rows; each row = list of (arr|None, caption). Uniform ncols within a montage."""
    ncols = max(len(r) for r in rows); pad = 8; cap = 20
    LG = 110 if row_labels else 0
    TB = 30 + (20 if col_headers else 0)
    cellH = cap + SIZE
    W = LG + ncols * SIZE + (ncols + 1) * pad
    H = TB + len(rows) * (cellH + pad) + pad
    canvas = Image.new("RGB", (W, H), (14, 15, 19)); d = ImageDraw.Draw(canvas)
    d.text((pad, 7), super_title, fill=(255, 230, 120), font=F_TITLE)
    if col_headers:
        for c, ch in enumerate(col_headers):
            d.text((LG + pad + c * (SIZE + pad) + 2, 32), ch, fill=(150, 200, 255), font=F_HEAD)
    for r, row in enumerate(rows):
        y = TB + r * (cellH + pad)
        if row_labels:
            d.text((6, y + cellH // 2 - 6), row_labels[r], fill=(230, 230, 230), font=F_HEAD)
        for c, (arr, capt) in enumerate(row):
            x = LG + pad + c * (SIZE + pad)
            if arr is not None:
                canvas.paste(Image.fromarray(arr.astype(np.uint8)).resize((SIZE, SIZE)), (x, y + cap))
            else:
                d.rectangle([x, y + cap, x + SIZE, y + cap + SIZE], fill=(50, 30, 30))
            d.text((x + 2, y + 4), capt, fill=(235, 235, 235), font=F_CAP)
    canvas.save(out_path); print(f"  wrote {out_path}  ({W}x{H}, {len(rows)} rows)", flush=True)


# ============ Point 1: brightfield preprocessing ============
def point1():
    datasets = [("branching", f"{KS}/Final_folder_uniform_fixedseed", "*.TIF"),
                ("pakp",      f"{KS}/EmrahPaKp_dataset_renamed",        "*.TIF"),
                ("nlev",      f"{KS}/NL_evolution_library_Image",       "*.TIF"),
                ("selx",      f"{KS}/Selected_Exps",                    "*.TIF")]
    rows = []
    for name, d, pat in datasets:
        fs = glob_sorted(d, pat); assert fs, f"no images in {d}"
        for fp in [fs[i] for i in even_idx(len(fs), 2)]:
            his = preprocess_experimental_initialstage_mod(fp)
            rows.append([(load_fit(fp), f"{name}  {os.path.basename(fp)}"),
                         (ours_dishremove(fp), "ours (black, x0.80)"),
                         (his, "kinshuk mod" if his is not None else "NO CIRCLE FOUND")])
    grid(rows, f"{OUT}/1_brightfield_preprocess.png",
         "Point 1 - brightfield preprocessing: adopt Kinshuk's preprocess_experimental_initialstage_mod",
         col_headers=["RAW (source)", "OURS - dish_remove", "KINSHUK - initialstage_mod (white, -82px, +contrast)"])


# ============ Point 2: mux / KL fluorescence spread (kept as-is) ============
def point2():
    mods = [("mux", MPX, "*.TIF"), ("kl", f"{KS}/KL_automatedcrops_2species_filtered_renamed", "*.jpg")]
    rows, labels = [], []
    for name, d, pat in mods:
        fs = sorted(glob_sorted(d, pat), key=mean_intensity); assert fs, f"no images in {d}"
        picks = [fs[i] for i in even_idx(len(fs), 6)]
        rows.append([(load_stretch(fp), f"mean={mean_intensity(fp):.0f}") for fp in picks])
        labels.append(f"{name}\n(n={len(fs)})")
    grid(rows, f"{OUT}/2_mux_kl_fluorescence.png",
         "Point 2 - mux/KL: train on NORMAL images now (intensity = label, NOT normalized). Sorted dim -> bright.",
         row_labels=labels)


# ============ Point 3: rattray (no crop / no inscribe) ============
def point3():
    fs = glob_sorted(f"{EXT}/rattray_2023/images", "*.png"); assert fs, "no rattray images"
    rows = [[(load_fit(fp), f"raw  {os.path.basename(fp)}"),
             (load_stretch(fp), "used: resize only, NO crop/inscribe")]
            for fp in [fs[i] for i in even_idx(len(fs), 4)]]
    grid(rows, f"{OUT}/3_rattray.png",
         "Point 3 - rattray: originals used as-is. No circular crop, no inscribe. Rough edge preserved.",
         col_headers=["RAW (native aspect)", "USED (256, no crop)"])


# ============ Point 4: swarmevo (dish masked) / myxo (normal, no mask) ============
# swarmevo frames are 256px, so Kinshuk's fixed radius-82 crop over-crops -> use our
# resolution-safe dish_remove (fractional x0.80). myxo has no dish -> normal.
def point4():
    swv = sorted(glob.glob(f"{EXT}/swarmevo/canon/*/*.png"))
    myx = sorted(glob.glob(f"{EXT}/myxococcus/images/**/*.jpg", recursive=True))
    assert swv and myx, "missing swarmevo or myxo frames"
    rows, labels = [], []
    for fp in [swv[i] for i in even_idx(len(swv), 3)]:
        rows.append([(load_fit(fp), f"raw  {os.path.basename(fp)}"),
                     (ours_dishremove(fp), "dish_remove (mask, x0.80)")])
        labels.append("swarmevo")
    for fp in [myx[i] for i in even_idx(len(myx), 3)]:
        rows.append([(load_fit(fp), f"raw  {os.path.basename(fp)}"),
                     (load_stretch(fp), "normal, no mask")])
        labels.append("myxo")
    grid(rows, f"{OUT}/4_swarmevo_myxo.png",
         "Point 4 - swarmevo: dish masked (resolution-safe dish_remove); myxo: normal, no mask.",
         col_headers=["RAW (native aspect)", "USED (256)"], row_labels=labels)


# ============ Compact: ONLY the final preprocessed inputs (no raw/before cols) ============
def compact():
    tiles = []
    for name, d, k in [("branching", f"{KS}/Final_folder_uniform_fixedseed", 2),
                       ("pakp", f"{KS}/EmrahPaKp_dataset_renamed", 1),
                       ("nlev", f"{KS}/NL_evolution_library_Image", 1),
                       ("selx", f"{KS}/Selected_Exps", 1)]:
        fs = glob_sorted(d, "*.TIF")
        for fp in [fs[i] for i in even_idx(len(fs), k)]:
            tiles.append((preprocess_experimental_initialstage_mod(fp), f"{name} - initialstage_mod"))
    for name, d, pat in [("mux", MPX, "*.TIF"), ("kl", f"{KS}/KL_automatedcrops_2species_filtered_renamed", "*.jpg")]:
        fs = sorted(glob_sorted(d, pat), key=mean_intensity)
        for fp in [fs[i] for i in even_idx(len(fs), 2)]:
            tiles.append((load_stretch(fp), f"{name} - normal (mean={mean_intensity(fp):.0f})"))
    fs = glob_sorted(f"{EXT}/rattray_2023/images", "*.png")
    for fp in [fs[i] for i in even_idx(len(fs), 2)]:
        tiles.append((load_stretch(fp), "rattray - no crop"))
    fs = sorted(glob.glob(f"{EXT}/swarmevo/canon/*/*.png"))
    for fp in [fs[i] for i in even_idx(len(fs), 3)]:
        tiles.append((ours_dishremove(fp), "swarmevo - dish_remove"))
    ncols = 5
    rows = [tiles[i:i + ncols] for i in range(0, len(tiles), ncols)]
    while len(rows[-1]) < ncols:
        rows[-1].append((None, ""))
    grid(rows, f"{OUT}/kinshuk_final_inputs.png",
         "Preprocessed training inputs - Kinshuk 4-point feedback applied (2026-11-16)")


# ============ Runtime-augmentation preview (prototype of the dataloader aug) ============
def _rot(img, angle, fill):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

def _photo(img, bright, contrast):
    f = (img.astype(np.float32) - 128) * contrast + 128
    f *= bright
    return np.clip(f, 0, 255).astype(np.uint8)

def augment_pair(src, tgt, rng, rotate=True, flip=True, brightness=True, fill=(255, 255, 255)):
    """SAME geometric + photometric params applied to BOTH source and target (keeps the pair aligned).
    brightness=False for fluorescent (mux/kl) -- intensity is the label."""
    ang = float(rng.uniform(0, 360)) if rotate else 0.0
    flp = bool(rng.random() < 0.5) if flip else False
    br = float(rng.uniform(0.85, 1.15)) if brightness else 1.0
    ct = float(rng.uniform(0.85, 1.15)) if brightness else 1.0
    out = []
    for im in (src, tgt):
        a = _rot(im, ang, fill) if rotate else im
        if flp:
            a = np.ascontiguousarray(a[:, ::-1])
        if brightness:
            a = _photo(a, br, ct)
        out.append(a)
    return out[0], out[1], f"rot={ang:.0f} flip={'T' if flp else 'F'} b={br:.2f} c={ct:.2f}"

def aug_preview():
    import collections
    fs = glob_sorted(f"{KS}/Final_folder_uniform_fixedseed", "*.TIF")
    byc = collections.OrderedDict()
    for fp in fs:
        byc.setdefault(os.path.basename(fp).rsplit("_", 1)[0], []).append(fp)
    conds = [c for c in byc if len(byc[c]) >= 2][:3]
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for c in conds:
        ps = preprocess_experimental_initialstage_mod(byc[c][0])
        pt = preprocess_experimental_initialstage_mod(byc[c][1])
        if ps is None or pt is None:
            continue
        for _ in range(2):
            asrc, atgt, lab = augment_pair(ps, pt, rng, fill=(255, 255, 255))
            rows.append([(ps, "src preproc"), (pt, "tgt preproc"), (asrc, "src+aug"), (atgt, "tgt+aug")])
            labels.append(c + "\n" + lab)
    grid(rows, f"{OUT}/5_branching_aug_preview.png",
         "Branching - runtime augmentation preview (same rot/flip/brightness to source+target; white fill)",
         col_headers=["source (hint) preproc", "target preproc", "source + AUG", "target + AUG"],
         row_labels=labels)


if __name__ == "__main__":
    print("building review montages ->", OUT, flush=True)
    point1(); point2(); point3(); point4(); compact()
    print("done.", flush=True)
