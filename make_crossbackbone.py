"""Cross-backbone, cross-domain viewer montages. Rows = conditions (deterministic evenly-spaced,
non-cherry-picked), cols = source | real | each backbone's best config. Two domains show the verdict
FLIP: SD1.5 wins branching / PixCell washed; PixCell wins multiplexed. Verified CMMD in labels."""
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

GRN=(134,239,172); BLUE=(125,211,252); CORAL=(252,165,165); AMBER=(251,191,36); WHITE=(224,224,224); GRAY=(150,150,150)

DOMAINS = {
 "branching": {
   "title": "BRANCHING colonies (dendritic)  —  SD1.5 wins, PixCell washes out",
   "src": f"{DATA}/branching_256/test/sources/{{c}}_src.TIF",
   "real": f"{DATA}/branching_256/test/reals/{{c}}_Rep*.TIF",
   "cols": [
     ("SD1.5 in-domain", "CMMD 3.4", f"{R}/sd15_branching_eval/{{c}}_1.png", GRN),
     ("Flux bf16 (best)", "CMMD 4.8", f"{R}/flux_cn_runs/scaled_bf16/eval_cn10/{{c}}_1.png", BLUE),
     ("PixCell-256", "CMMD 49.8", f"{R}/pixcell_gen/branching256_ep30_ddpm/{{c}}_1.png", CORAL),
     ("PixCell-1024", "CMMD 55.1", f"{R}/pixcell1024_gen/branching_ddpm/{{c}}_1.png", CORAL),
   ],
   "conds_from": f"{DATA}/branching_256/test/sources/*_src.TIF",
 },
 "multiplexed": {
   "title": "MULTIPLEXED colonies (cell-like, 2-gene patterning)  —  PixCell wins, Flux N/A (never trained here)",
   "src": f"{DATA}/multiplexed_eval/sources/{{c}}_src.TIF",
   "real": f"{DATA}/multiplexed_eval/reals_heldout/{{c}}_Rep*.TIF",
   "cols": [
     ("SD1.5 in-domain", "CMMD 12.6", f"{DATA}/multiplexed_eval/synth_ft/{{c}}_*_s0.png", GRN),
     ("SD1.5 generalist", "", f"{DATA}/multiplexed_eval/synth_generalist/{{c}}_*_s0.png", AMBER),
     ("PixCell ep30 (best)", "CMMD 5.8", f"{R}/pixcell_eval_out/pixcell_ep30/gen/{{c}}_1.png", CORAL),
   ],
   "conds_from": f"{DATA}/multiplexed_eval/sources/*_src.TIF",
 },
}

def first(pat):
    g = sorted(glob.glob(pat)); return g[0] if g else None

def build(key, NC=4, TILE=240):
    cfg = DOMAINS[key]
    conds = sorted(os.path.basename(s).split("_")[0] for s in glob.glob(cfg["conds_from"]))
    CONDS = [conds[int(i*(len(conds)-1)/(NC-1))] for i in range(NC)]
    COLS = [("source","(input)",cfg["src"],WHITE), ("real #1","(target)",cfg["real"],WHITE)] + \
           [(a,b,c,d) for (a,b,c,d) in cfg["cols"]]
    GAP, LAB, HEAD, TITLE_H = 6, 56, 46, 40
    W = LAB + len(COLS)*(TILE+GAP); H = TITLE_H + HEAD + len(CONDS)*(TILE+GAP)
    im = Image.new("RGB", (W, H), (11,12,15)); d = ImageDraw.Draw(im)
    fT, fH, fh, fR = font(22,True), font(15,True), font(12), font(12,True)
    d.text((LAB-8, 10), cfg["title"], font=fT, fill=WHITE)
    for ci,(a,b,_,col) in enumerate(COLS):
        cx = LAB + ci*(TILE+GAP)
        d.text((cx, TITLE_H+6), a, font=fH, fill=col)
        if b: d.text((cx, TITLE_H+26), b, font=fh, fill=GRAY)
    y0 = TITLE_H + HEAD
    for ri,cond in enumerate(CONDS):
        ry = y0 + ri*(TILE+GAP)
        d.text((4, ry+TILE//2-6), cond, font=fR, fill=BLUE)
        for ci,(a,b,pat,col) in enumerate(COLS):
            cx = LAB + ci*(TILE+GAP)
            fp = first(pat.format(c=cond))
            if fp:
                im.paste(Image.open(fp).convert("RGB").resize((TILE,TILE), Image.LANCZOS), (cx, ry))
            else:
                d.rectangle([cx,ry,cx+TILE,ry+TILE], outline=(70,70,76)); d.text((cx+8,ry+8),"—",font=fh,fill=CORAL)
    out = f"{R}/cross_{key}.png"; im.save(out); print("saved", out, im.size)

build("branching"); build("multiplexed")
