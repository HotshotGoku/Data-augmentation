"""THE difference, visual + clean. For each colony: source | real sibling | ROTATION x3 (classical aug) |
GENERATIVE synth x3 (conditioning-dropout). Shows: rotation copies = same colony rotated (identical
shape -> 0 new morphology info); generative synth = genuinely different shapes -> new shape_factor labels."""
import os, glob, cv2
from PIL import Image, ImageDraw, ImageFont
D = "/hpc/group/youlab/sa603/data/branching_256"
DROP = "/hpc/group/youlab/sa603/data/branching_downstream/synth_train_dropout05g9"

def font(sz, bold=False):
    for p in [f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()
def load(fp, t=220): return cv2.resize(cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (t, t))
def rot(img, a):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w/2, h/2), a, 1.0), (w, h), borderValue=(0,0,0))

conds = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{DROP}/*_Rep1_s0.png")})
conds = [c for c in conds if os.path.exists(f"{D}/canon/{c}_Rep1.TIF") and os.path.exists(f"{D}/canon/{c}_Rep2.TIF")]
SEL = [conds[int(i*(len(conds)-1)/3)] for i in range(4)]

WHITE=(235,235,235); AMBER=(251,191,36); GRN=(134,239,172); GRAY=(150,150,150)
TILE, GAP, LAB, TITLE_H, GRP_H = 220, 7, 70, 82, 30
NCOL = 8  # source, real, 3 rot, 3 drop
W = LAB + NCOL*(TILE+GAP); H = TITLE_H + GRP_H + 26 + len(SEL)*(TILE+GAP) + 64
im = Image.new("RGB", (W, H), (11,12,15)); d = ImageDraw.Draw(im)
d.text((LAB-10, 12), "The difference: classical rotation vs generative augmentation", font=font(25,True), fill=WHITE)
d.text((LAB-10, 48), "Rotation makes copies of the SAME colony (identical shape).  Generative synth makes genuinely DIFFERENT colonies (new shapes).", font=font(14), fill=GRAY)

def gx(ci): return LAB + ci*(TILE+GAP)
# group header bars
d.rectangle([gx(2)-3, TITLE_H, gx(5)-GAP, TITLE_H+GRP_H-6], fill=(40,33,10))
d.text((gx(2)+4, TITLE_H+5), "CLASSICAL AUGMENTATION  (rotation — same shape, 0 new morphology info)", font=font(13,True), fill=AMBER)
d.rectangle([gx(5)-3, TITLE_H, gx(8)-GAP, TITLE_H+GRP_H-6], fill=(12,38,20))
d.text((gx(5)+4, TITLE_H+5), "GENERATIVE AUGMENTATION  (conditioning-dropout — new shapes = new labels)", font=font(13,True), fill=GRN)
for ci,name in enumerate(["source","real sibling"]):
    d.text((gx(ci), TITLE_H+8), name, font=font(13,True), fill=WHITE)

y0 = TITLE_H + GRP_H + 26
for ri,c in enumerate(SEL):
    ry = y0 + ri*(TILE+GAP)
    d.text((6, ry+TILE//2-6), c, font=font(13,True), fill=(130,200,250))
    src = load(f"{D}/canon/{c}_Rep1.TIF")
    drs = sorted(glob.glob(f"{DROP}/{c}_Rep1_s*.png"))[:3]
    tiles = [src, load(f"{D}/canon/{c}_Rep2.TIF"), rot(src,100), rot(src,190), rot(src,280)] + [load(x) for x in drs]
    for ci,t in enumerate(tiles[:NCOL]):
        im.paste(Image.fromarray(t), (gx(ci), ry))
cy = y0 + len(SEL)*(TILE+GAP) + 8
d.text((LAB-10, cy), "Downstream @N60 (15-seed, paired): generative shape_factor 0.396 vs rotation 0.339  (+0.057, p<0.05, 12/15 seeds).", font=font(14,True), fill=GRN)
d.text((LAB-10, cy+22), "Note: this diversity config also shifts color/contrast (a realism cost) — but shape_factor ignores color; the SHAPE variation is the signal.", font=font(13), fill=GRAY)
out = "/hpc/group/youlab/sa603/code/Data_augmentation/difference_dropout_vs_rotation.png"
im.save(out); print("saved", out, im.size, SEL)
