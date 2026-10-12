"""Before/after realism: real colonies vs OLD diversity config (dropout s0.5 — too big/bright) vs
NEW config (dropout s1.0/guidance7 — realistic AND still diverse)."""
import os, glob, cv2
from PIL import Image, ImageDraw, ImageFont
D = "/hpc/group/youlab/sa603/data/branching_256"
SW = "/hpc/group/youlab/sa603/code/Data_augmentation/dropout_realsweep"
OLD = f"{SW}/drop_s0.5_g9.0"; NEW = f"{SW}/drop_s1.0_g7.0"
def F(s,b=False):
    try: return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if b else ''}.ttf", s)
    except Exception: return ImageFont.load_default()
def L(fp,t=220): return cv2.resize(cv2.cvtColor(cv2.imread(fp,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB),(t,t))
conds = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{NEW}/*.png")})
conds = [c for c in conds if os.path.exists(f"{D}/test/reals/{c}_Rep1.TIF")]
SEL = [conds[int(i*(len(conds)-1)/3)] for i in range(4)]
WHITE=(235,235,235); CORAL=(252,165,165); GRN=(134,239,172); GRAY=(150,150,150)
TILE,GAP,LAB,TH,GH = 220,7,70,82,30
COLS=["source","real #1","real #2"]; W=LAB+7*(TILE+GAP); H=TH+GH+26+len(SEL)*(TILE+GAP)+44
im=Image.new("RGB",(W,H),(11,12,15)); d=ImageDraw.Draw(im)
d.text((LAB-10,12),"Fixing realism: OLD diversity config vs NEW (dropout at full strength)",font=F(24,True),fill=WHITE)
d.text((LAB-10,48),"Turning conditioning strength back up (1.0) anchors size + brightness to the real colony — while the dropout training keeps diversity.",font=F(13),fill=GRAY)
def gx(c): return LAB+c*(TILE+GAP)
d.rectangle([gx(3)-3,TH,gx(5)-GAP,TH+GH-6],fill=(45,15,15)); d.text((gx(3)+4,TH+5),"OLD  (dropout s0.5 — too big/bright)  realism 0.40",font=F(12,True),fill=CORAL)
d.rectangle([gx(5)-3,TH,gx(7)-GAP,TH+GH-6],fill=(12,38,20)); d.text((gx(5)+4,TH+5),"NEW  (dropout s1.0/g7 — realistic + diverse)  realism 0.33",font=F(12,True),fill=GRN)
for c,n in enumerate(COLS): d.text((gx(c),TH+8),n,font=F(13,True),fill=WHITE)
y0=TH+GH+26
for ri,c in enumerate(SEL):
    ry=y0+ri*(TILE+GAP); d.text((6,ry+TILE//2-6),c,font=F(13,True),fill=(130,200,250))
    old=sorted(glob.glob(f"{OLD}/{c}_*.png"))[:2]; new=sorted(glob.glob(f"{NEW}/{c}_*.png"))[:2]
    tiles=[L(f"{D}/test/sources/{c}_src.TIF"),L(f"{D}/test/reals/{c}_Rep1.TIF"),L(f"{D}/test/reals/{c}_Rep2.TIF")]+[L(x) for x in old]+[L(x) for x in new]
    for ci,t in enumerate(tiles[:7]): im.paste(Image.fromarray(t),(gx(ci),ry))
cy=y0+len(SEL)*(TILE+GAP)+8
d.text((LAB-10,cy),"NEW config: realism 0.333 (~baseline 0.324), CMMD 3.20 (best of all), diversity 0.271 (>baseline 0.252). Right size, right brightness, still varied.",font=F(13,True),fill=GRN)
out="/hpc/group/youlab/sa603/code/Data_augmentation/realism_old_vs_new.png"; im.save(out); print("saved",out,im.size,SEL)
