"""Multiplexed bake-off comparison: is the PixCell CMMD win (5.8 vs 12.6) visible?
Rows = conditions; cols = Source | Real#1 | Real#2 | SD1.5 mplex-ft | PixCell-256.
PixCell metrics read from the freshly-scored metrics.json so labels match the exact images shown.
"""
import os, glob, json
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
M = f"{BASE}/mplex"          # staging: sources/ reals/ sd15/ pixcell/ metrics.json
CONDS = ["R1C1", "R5C4", "R10C7"]
TILE, GAP, LG = 300, 12, 108
TITLE_H, HEAD_H, CAP_H = 104, 92, 78

# SD1.5 mplex-ft reference numbers (bake-off table); PixCell read live from metrics.json
SD = dict(cmmd="12.6", real="0.412", div="0.381")
pc = dict(cmmd="5.8", real="0.346", div="0.259")  # fallback = bake-off values
mj = f"{M}/metrics.json"
if os.path.exists(mj):
    o = json.load(open(mj)).get("overall", {})
    g = lambda k: (o[k]["mean"] if isinstance(o.get(k), dict) else o.get(k))
    try: pc = dict(cmmd=f"{g('CMMD'):.1f}", real=f"{g('realism_vs_sibling'):.3f}", div=f"{g('diversity'):.3f}")
    except Exception: pass

def font(sz, bold=False):
    for p in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

def load(path):
    return Image.open(path).convert("RGB").resize((TILE, TILE), Image.LANCZOS)

def first(pat):
    fs = sorted(glob.glob(pat)); return fs[0] if fs else None

COLS = [
    ("Source", "(input replicate)", None),
    ("Real replicate #1", "", None),
    ("Real replicate #2", "", None),
    ("SD1.5 mplex-ft", f"CMMD {SD['cmmd']} · real {SD['real']} · div {SD['div']}", "sd15"),
    ("PixCell-256", f"CMMD {pc['cmmd']} · real {pc['real']} · div {pc['div']}", "pixcell"),
]
NC = len(COLS)
W = LG + NC * TILE + (NC - 1) * GAP + 16
H = TITLE_H + HEAD_H + len(CONDS) * (TILE + GAP) + CAP_H
canvas = Image.new("RGB", (W, H), (14, 15, 19))
d = ImageDraw.Draw(canvas)
fT, fH, fh, fR = font(23, True), font(15, True), font(12), font(13, True)
TITLE, GRAY, GREEN, CORAL, BLUE = (233,234,236), (154,160,166), (134,239,172), (252,165,165), (125,211,252)

d.text((LG-8, 14), "Multiplexed domain — PixCell WINS on CMMD (5.8 vs SD1.5's 12.6). Is that ~2x distribution gain visible per-image?", font=fT, fill=TITLE)
d.text((LG-8, 46), "CMMD = distribution-level CLIP distance, lower=better (only comparable within a domain). Realism ÷ 0.32 floor, lower=better. Diversity=LPIPS among own samples, higher=better.", font=fh, fill=GRAY)
d.text((LG-8, 66), "Opposite of branching: here PixCell's cell-microscopy prior FITS the cell-like dishes, so it's more realistic — but note its lower diversity (the tradeoff), which is what gates downstream.", font=fh, fill=BLUE)

x0 = LG
for ci, (t1, t2, key) in enumerate(COLS):
    cx = x0 + ci * (TILE + GAP)
    col = TITLE if ci < 3 else (GREEN if key == "pixcell" else CORAL)
    d.text((cx, TITLE_H + 8), t1, font=fH, fill=col)
    if t2: d.text((cx, TITLE_H + 30), t2, font=fh, fill=GRAY)

y0 = TITLE_H + HEAD_H
for ri, cond in enumerate(CONDS):
    ry = y0 + ri * (TILE + GAP)
    d.text((6, ry + TILE//2 - 8), cond, font=fR, fill=BLUE)
    reals = sorted(glob.glob(f"{M}/reals/{cond}_Rep*.TIF")) or sorted(glob.glob(f"{M}/reals/{cond}_Rep*.png"))
    tiles = [
        first(f"{M}/sources/{cond}_src.*"),
        reals[0] if reals else None,
        reals[1] if len(reals) > 1 else None,
        first(f"{M}/sd15/{cond}_*"),
        first(f"{M}/pixcell/{cond}_*"),
    ]
    for ci, tp in enumerate(tiles):
        cx = x0 + ci * (TILE + GAP)
        if tp and os.path.exists(tp):
            canvas.paste(load(tp), (cx, ry))
        else:
            d.rectangle([cx, ry, cx+TILE, ry+TILE], outline=(60,60,66)); d.text((cx+8, ry+8), "missing", font=fh, fill=CORAL)

cy = y0 + len(CONDS) * (TILE + GAP) + 6
d.text((LG-8, cy), f"Takeaway: on cell-like multiplexed dishes PixCell's prior helps — CMMD {pc['cmmd']} vs {SD['cmmd']}, realism {pc['real']} vs {SD['real']} (both better).", font=fh, fill=GREEN)
d.text((LG-8, cy+20), f"But PixCell's diversity is LOWER ({pc['div']} vs {SD['div']}) — the realism/diversity tradeoff — and downstream is gated by diversity, so PixCell did not beat SD1.5 downstream despite looking more realistic.", font=fh, fill=GRAY)
d.text((LG-8, cy+40), "This is the crux for Kinshuk's point 1: per-image realism improved, yet the metric that matters for downstream (diversity) got worse. A better backbone would need to lift BOTH.", font=fh, fill=GRAY)

out = f"{BASE}/multiplexed_bakeoff_comparison.png"
canvas.save(out); print("saved", out, canvas.size, "| PixCell scored:", pc)
