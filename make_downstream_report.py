"""Compare the downstream sim->exp experiment: A (real-only) vs B (real+synthetic).
Builds a metrics table (SSIM / LPIPS) + prediction montages (sim input | real exp | A pred | B pred)
+ an HTML index. Reads the eval outputs written by simtoexp_eval_ds.py. cv2 only (no model).
Env: DS_A_TAG (default real_N50), DS_B_TAG (default realsynth_N50). Upload to Data_augmentation root.
"""
import os
import glob
import json
import cv2
import numpy as np
import Data_augmentation.utils.shared_resources_config  # noqa: F401 — puts cldm on sys.path (import BEFORE cldm)
from cldm.preprocess import preprocess_simulation_graybackground, preprocess_experimental_backgroundblack

REPO = "/hpc/group/youlab/sa603/code/Data_augmentation"
A = os.environ.get("DS_A_TAG", "real_N50")
B = os.environ.get("DS_B_TAG", "realsynth_N50")
EVAL = os.path.join(REPO, "downstream_eval")
SIM_TEST = "/hpc/group/youlab/sa603/data/sim_project/extracted/sim_to_exp_diffusion/SimcorrtoExp_testset"
EXP_TEST = "/hpc/group/youlab/sa603/data/sim_project/extracted/sim_to_exp_diffusion/Exp_testset"
OUT = os.path.join(REPO, "downstream_report")
os.makedirs(OUT, exist_ok=True)
CELL = 256


def load(p):
    if not p or not os.path.exists(p):
        return None
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return None if im is None else cv2.resize(im, (CELL, CELL))


def strip(imgs, labels, title):  # all cells BGR
    lab = 22
    cells = []
    for im, l in zip(imgs, labels):
        c = np.full((CELL + lab, CELL, 3), 30, np.uint8)
        if im is not None:
            c[lab:] = im
        cv2.putText(c, str(l)[:32], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(c)
    sep = np.full((cells[0].shape[0], 4, 3), 30, np.uint8)
    row = np.hstack([x for c in cells for x in (c, sep)])
    ban = np.full((26, row.shape[1], 3), 15, np.uint8)
    cv2.putText(ban, title, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([ban, row])


Apred, Bpred = os.path.join(EVAL, A, "pred"), os.path.join(EVAL, B, "pred")
names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(Apred, "*.png")))[:6]
rows = []
for n in names:
    tif = n.replace(".png", ".TIF")
    sim = preprocess_simulation_graybackground(os.path.join(SIM_TEST, tif))
    sim = np.repeat(sim[:, :, None], 3, 2).astype(np.uint8)  # gray -> BGR==RGB
    real = preprocess_experimental_backgroundblack(os.path.join(EXP_TEST, tif))  # RGB
    rows.append(strip(
        [sim, cv2.cvtColor(real, cv2.COLOR_RGB2BGR), load(os.path.join(Apred, n)), load(os.path.join(Bpred, n))],
        ["sim input", "real exp", f"A: {A}", f"B: {B}"], f"test {n}"))
if rows:
    gap = np.full((6, rows[0].shape[1], 3), 0, np.uint8)
    cv2.imwrite(os.path.join(OUT, "predictions.png"), np.vstack([x for r in rows for x in (r, gap)]))
    print("wrote predictions.png")


def metrics(tag):
    p = os.path.join(EVAL, tag, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}


ma, mb = metrics(A), metrics(B)
delta_ssim = (mb.get("SSIM_mean") or 0) - (ma.get("SSIM_mean") or 0)
delta_lpips = (mb.get("LPIPS_mean") or 0) - (ma.get("LPIPS_mean") or 0)
html = f"""<html><head><meta charset='utf-8'></head>
<body style='font-family:sans-serif;background:#111;color:#eee;max-width:1300px;margin:auto'>
<h1>Downstream utility: sim&rarr;exp, real-only (A) vs real+synthetic (B)</h1>
<p>Held-out real Exp_testset. SSIM higher = better; LPIPS lower = better. B&minus;A: SSIM {delta_ssim:+.4f}, LPIPS {delta_lpips:+.4f} (LPIPS negative = synthetic helped).</p>
<table border=1 cellpadding=6 style='border-collapse:collapse'>
<tr><th>condition</th><th>SSIM &uarr;</th><th>LPIPS &darr;</th><th>n</th></tr>
<tr><td>A &mdash; {A} (real-only)</td><td>{ma.get('SSIM_mean')}</td><td>{ma.get('LPIPS_mean')}</td><td>{ma.get('n')}</td></tr>
<tr><td>B &mdash; {B} (real+synthetic)</td><td>{mb.get('SSIM_mean')}</td><td>{mb.get('LPIPS_mean')}</td><td>{mb.get('n')}</td></tr>
</table>
<h2>Predictions (sim input | real exp | A | B)</h2><img src='predictions.png' style='max-width:100%'>
</body></html>"""
open(os.path.join(OUT, "index.html"), "w").write(html)
print("wrote", os.path.join(OUT, "index.html"))
