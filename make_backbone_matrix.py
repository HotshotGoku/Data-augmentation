"""Unified backbone x dataset matrix. ROWS = datasets, COLS = every model we tried.
One representative, condition-aligned sample per dataset row; 'not tried' tile where a backbone
was never run on that dataset. Companion to make_crossbackbone.py (which does per-domain montages)."""
import os, glob
from PIL import Image, ImageDraw, ImageFont
R = "/hpc/group/youlab/sa603/code/Data_augmentation"
DATA = "/hpc/group/youlab/sa603/data"

def font(sz, bold=False):
    for p in [f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

WHITE=(224,224,224); GRN=(134,239,172); BLUE=(125,211,252); CORAL=(252,165,165)
AMBER=(251,191,36); GRAY=(150,150,150); VIOLET=(196,181,253)

def first(pat):
    g = sorted(glob.glob(pat)); return g[0] if g else None

# (label, color, {dataset: path-pattern})  — pattern may contain {c}; unconditional uses a fixed file.
COLS = [
 ("source\n(input)", WHITE, {
    "branching":   f"{DATA}/branching_256/test/sources/{{c}}_src.TIF",
    "multiplexed": f"{DATA}/multiplexed_eval/sources/{{c}}_src.TIF"}),
 ("real\n(target)", WHITE, {
    "branching":   f"{DATA}/branching_256/test/reals/{{c}}_Rep*.TIF",
    "multiplexed": f"{DATA}/multiplexed_eval/reals_heldout/{{c}}_Rep*.TIF"}),
 ("SD1.5\nin-domain", GRN, {
    "branching":   f"{R}/sd15_branching_eval/{{c}}_1.png",
    "multiplexed": f"{DATA}/multiplexed_eval/synth_ft/{{c}}_*_s0.png"}),
 ("SD1.5\ndropout*", GRN, {
    "branching":   f"{R}/dropout_realsweep/drop_s1.0_g7.0/{{c}}_1.png"}),
 ("SD1.5\ngeneralist", AMBER, {
    "branching":   f"{R}/generalist_eval_out/gen_epoch=5/ff/gen/{{c}}_1.png",
    "multiplexed": f"{DATA}/multiplexed_eval/synth_generalist/{{c}}_*_s0.png"}),
 ("Flux\nbf16+CN6", BLUE, {
    "branching":   f"{R}/flux_cn_runs/scaled_bf16/eval_cn10/{{c}}_1.png"}),
 ("PixCell", CORAL, {
    "branching":   f"{R}/pixcell_gen/branching256_ep30_ddpm/{{c}}_1.png",
    "multiplexed": f"{R}/pixcell_eval_out/pixcell_ep30/gen/{{c}}_1.png"}),
 ("FastGAN\n(uncond)", VIOLET, {
    "branching":   f"{R}/gan_runs/gen/lwgan_m2/gen_0003.png"}),
 ("conditional\nDDPM", CORAL, {
    "branching":   f"{R}/cond_ddpm_runs/gen250/{{c}}_1.png"}),
 ("zero-shot\nSDXL", CORAL, {
    "branching":   f"{R}/backbone_screen/sdxl_s70/{{c}}_1.png"}),
]

DATASETS = [
 ("branching",   "BRANCHING  (dendritic, petri dish)  ·  downstream = morphology features", "ffc112"),
 ("multiplexed", "MULTIPLEXED  (cell-like fluorescence, no dish)  ·  downstream = aTc / IPTG decode", "R10C1"),
]

def pick_cond(ds, prefer):
    src = COLS[0][2][ds]
    conds = sorted(os.path.basename(s).split("_")[0] for s in glob.glob(src.replace("{c}","*")))
    if prefer and prefer in conds: return prefer
    prim = COLS[2][2][ds]
    for c in conds:
        if first(prim.format(c=c)): return c
    return conds[0] if conds else None

TILE, GAP, LAB, HEAD, TITLE_H, ROWGAP = 258, 6, 132, 60, 44, 34
nC = len(COLS)
W = LAB + nC*(TILE+GAP)
rowH = TITLE_H + HEAD + TILE + ROWGAP
H = len(DATASETS)*rowH + 8
im = Image.new("RGB",(W,H),(11,12,15)); d = ImageDraw.Draw(im)
fT, fH, fh, fR = font(22,True), font(14,True), font(12), font(14,True)

for di,(ds,title,prefer) in enumerate(DATASETS):
    oy = di*rowH
    cond = pick_cond(ds, prefer)
    d.text((8, oy+12), title, font=fT, fill=WHITE)
    cy = oy+TITLE_H+HEAD+TILE//2
    d.text((8, cy-22), ds, font=fR, fill=BLUE)
    d.text((8, cy-2), f"cond {cond}", font=fh, fill=GRAY)
    for ci,(lab,col,pats) in enumerate(COLS):
        cx = LAB + ci*(TILE+GAP); hy = oy+TITLE_H
        for li,line in enumerate(lab.split("\n")):
            d.text((cx, hy+4+li*18), line, font=fH, fill=col)
        iy = oy+TITLE_H+HEAD
        pat = pats.get(ds); fp = None
        if pat: fp = first(pat.format(c=cond) if "{c}" in pat else pat)
        if fp:
            im.paste(Image.open(fp).convert("RGB").resize((TILE,TILE),Image.LANCZOS),(cx,iy))
        else:
            d.rectangle([cx,iy,cx+TILE,iy+TILE], fill=(20,21,26), outline=(60,60,66))
            d.text((cx+TILE//2-42, iy+TILE//2-8), "not tried", font=fh, fill=(120,120,126))
out = f"{R}/backbone_matrix.png"; im.save(out); print("saved", out, im.size)
