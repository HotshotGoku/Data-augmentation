"""Assemble example-image montages + an HTML report from the augmenter experiments — so every
result is documented visually, not just as numbers. Reads ALREADY-generated images (cv2 only, no
model): the OOD fine-tune comparison (baseline vs freeze=none vs freeze=shallow vs real sibling)
and the inference scale x strength sweeps (OOD + in-domain). Run on a compute node.

Upload to repo root; writes results_report/ (montages + index.html)."""
import os
import glob
import csv
import cv2
import numpy as np

REPO = os.environ.get("AUG_REPO", "/hpc/group/youlab/sa603/code/Data_augmentation")
TEST2SP = "/hpc/group/youlab/sa603/data/finetune_2sp/test_2sp"
INDOMAIN = "/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3"
SWEEP = os.path.join(REPO, "sweep_eval_2sp")
INFER = os.path.join(REPO, "infer_sweep_out")
OUT = os.path.join(REPO, "results_report")
os.makedirs(OUT, exist_ok=True)
CELL = 256


def load(path, size=CELL):
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if img is None else cv2.resize(img, (size, size))


def strip(imgs, labels, title):
    lab_h, title_h, pad = 22, 26, 4
    cells = []
    for im, lab in zip(imgs, labels):
        c = np.full((CELL + lab_h, CELL, 3), 30, np.uint8)
        if im is not None:
            c[lab_h:, :, :] = im
        cv2.putText(c, str(lab)[:34], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(c)
    sep = np.full((cells[0].shape[0], pad, 3), 30, np.uint8)
    row = np.hstack([x for c in cells for x in (c, sep)])
    banner = np.full((title_h, row.shape[1], 3), 15, np.uint8)
    cv2.putText(banner, title, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, row])


def prefix_files(folder, ext="*.TIF"):
    m = {}
    for fp in sorted(glob.glob(os.path.join(folder, ext))):
        m.setdefault(os.path.basename(fp).split("_")[0], []).append(fp)
    return m


def stack(rows):
    w = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == w else np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 0, np.uint8)]) for r in rows]
    gap = np.full((6, w, 3), 0, np.uint8)
    return np.vstack([x for r in rows for x in (r, gap)])


# ---- 1) Fine-tune OOD comparison: source | baseline | none e3 | shallow e3 | real sibling ----
srcmap = prefix_files(TEST2SP)
prefixes = sorted(srcmap.keys())[:6]
ft_tags = [("baseline_fulldata", "baseline"), ("none_lr5e-6_epoch=3", "ft none e3"),
           ("shallowsweep_lr5e-6_epoch=3", "ft shallow e3 (best)")]
rows = []
for pfx in prefixes:
    files = srcmap[pfx]
    imgs, labs = [load(files[0])], [f"source {pfx}"]
    for tag, short in ft_tags:
        imgs.append(load(os.path.join(SWEEP, tag, "gen", f"{pfx}_1.png")))
        labs.append(short)
    imgs.append(load(files[1] if len(files) > 1 else files[0]))
    labs.append("real sibling")
    rows.append(strip(imgs, labs, f"OOD colony {pfx}: baseline -> fine-tuned -> real"))
if rows:
    cv2.imwrite(os.path.join(OUT, "finetune_comparison_OOD.png"), stack(rows))
    print("wrote finetune_comparison_OOD.png")


# ---- 2) Inference scale x strength grids (OOD + in-domain), a couple prefixes each ----
def infer_grid(target, srcfolder, out_name):
    smap = prefix_files(srcfolder)
    pref = sorted(smap.keys())[:2]
    scales, strengths = ["9.0", "12.0", "15.1"], ["1.0", "1.5", "2.0"]
    blocks = []
    for pfx in pref:
        rws = [strip([load(smap[pfx][0])], [f"source {pfx}"], f"{target} colony {pfx} — scale x strength")]
        for sc in scales:
            imgs = [load(os.path.join(INFER, target, f"scale{sc}_str{st}", f"{pfx}_1.png")) for st in strengths]
            rws.append(strip(imgs, [f"scale{sc} str{st}" for st in strengths], f"scale {sc}"))
        blocks.append(stack(rws))
    if blocks:
        # pad to same width then stack
        w = max(b.shape[1] for b in blocks)
        blocks = [np.hstack([b, np.full((b.shape[0], w - b.shape[1], 3), 0, np.uint8)]) for b in blocks]
        cv2.imwrite(os.path.join(OUT, out_name), stack(blocks))
        print("wrote", out_name)


infer_grid("OOD", TEST2SP, "infersweep_OOD.png")
infer_grid("IND", INDOMAIN, "infersweep_IND.png")


# ---- 3) HTML index tying tables + montages together ----
def table(tsv):
    if not os.path.exists(tsv):
        return "<p>(missing)</p>"
    rs = list(csv.reader(open(tsv), delimiter="\t"))
    out = "<table border=1 cellpadding=4 style='border-collapse:collapse'>"
    for i, r in enumerate(rs):
        t = "th" if i == 0 else "td"
        out += "<tr>" + "".join(f"<{t}>{c}</{t}>" for c in r) + "</tr>"
    return out + "</table>"


imgs_html = ""
for name, cap in [("finetune_comparison_OOD.png", "Fine-tune closes the OOD gap (source → baseline → none → shallow → real)"),
                  ("infersweep_OOD.png", "Inference scale×strength on OOD (higher strength = worse)"),
                  ("infersweep_IND.png", "Inference scale×strength in-domain")]:
    if os.path.exists(os.path.join(OUT, name)):
        imgs_html += f"<h3>{cap}</h3><img src='{name}' style='max-width:100%'>"

html = f"""<html><head><meta charset='utf-8'><title>Augmenter results</title></head>
<body style='font-family:sans-serif;background:#111;color:#eee;max-width:1400px;margin:auto'>
<h1>Augmenter experiments — results & example images</h1>
<h2>Fine-tune sweep (OOD 2-species, lower realism = better)</h2>{table(os.path.join(SWEEP, 'summary.tsv'))}
<h2>Inference sweep (scale × strength)</h2>{table(os.path.join(INFER, 'summary.tsv'))}
<h2>Example images</h2>{imgs_html}
</body></html>"""
open(os.path.join(OUT, "index.html"), "w").write(html)
print("wrote", os.path.join(OUT, "index.html"))
