"""Constrain generated colonies into a FIXED, consistent petri dish. Build a clean dish template (mean
rim of real dishes), then for each generated image: scale its dish to the fixed size and composite only
its INTERIOR (the colony) inside the template's full rim. Result: every generated image = same-size dish,
full real rim, colony just inside. Fixes inconsistent/inflated/rim-less generated dishes."""
import cv2, numpy as np, glob, os
from PIL import Image
from dish_normalize import detect_dish
SIZE = 256; TARGET_R = 118
D = "/hpc/group/youlab/sa603/data/branching_256"
GEN = "/hpc/group/youlab/sa603/code/Data_augmentation/dropout_realsweep/drop_s1.0_g7.0"

def load(f): return np.array(Image.open(f).convert("RGB").resize((SIZE, SIZE)))
def scale_center(img, target_r=TARGET_R, size=SIZE):   # scale dish rim to fixed radius + center (NO mask)
    x, y, r = detect_dish(img)
    if not (70 < r < 140): x, y, r = size/2, size/2, target_r
    s = target_r / r; M = cv2.getRotationMatrix2D((x, y), 0, s); M[0,2] += size/2 - x; M[1,2] += size/2 - y
    return cv2.warpAffine(img, M, (size, size))

# fixed dish template = mean of many normalized REAL dishes (clean, consistent full rim)
reals = sorted(glob.glob(f"{D}/test/reals/*.TIF"))[:20]
template = np.mean([scale_center(load(f)).astype(np.float32) for f in reals], 0)

# radial feather: interior (colony, from GEN) -> rim/outer (from template), black outside dish
yy, xx = np.mgrid[0:SIZE, 0:SIZE]; rr = np.sqrt((xx - SIZE/2)**2 + (yy - SIZE/2)**2)
alpha = np.clip((112 - rr) / 10.0, 0, 1)[..., None]     # 1 inside r~102 (colony), 0 beyond r~112 (rim from template)
DISH = (rr <= 126)[..., None]

def composite(gen_img):
    g = scale_center(gen_img).astype(np.float32)
    out = alpha * g + (1 - alpha) * template            # colony from gen, full rim from template
    return (out * DISH).clip(0, 255).astype(np.uint8)

def batch(src_dir, dst_dir):
    """Standardize every image in src_dir -> dst_dir (consistent dish, full rim, colony inside). Preserves extension."""
    os.makedirs(dst_dir, exist_ok=True); n = 0
    for f in sorted(glob.glob(src_dir + "/*.png") + glob.glob(src_dir + "/*.TIF")):
        ext = os.path.splitext(os.path.basename(f))[1]
        Image.fromarray(composite(load(f))).save(os.path.join(dst_dir, os.path.splitext(os.path.basename(f))[0] + ext)); n += 1
    print(f"composited {n} -> {dst_dir}", flush=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:                     # batch mode: src_dir dst_dir
        batch(sys.argv[1], sys.argv[2]); sys.exit(0)
    # else demo montage: REAL | GEN original | GEN constrained-into-fixed-dish
    conds = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{GEN}/*.png")}
                   & {os.path.basename(f).split("_")[0] for f in glob.glob(f"{D}/test/reals/*.TIF")})
    SEL = [conds[int(i*(len(conds)-1)/3)] for i in range(4)]
    TILE = 240; cols = ["REAL (reference)", "GENERATED (original)", "GENERATED (fixed dish)"]
    colc = [(134,239,172),(252,165,165),(125,211,252)]
    from PIL import ImageDraw, ImageFont
    try: fB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception: fB = ImageFont.load_default()
    im = Image.new("RGB", (40+3*(TILE+8), 44+len(SEL)*(TILE+8)), (11,12,15)); d = ImageDraw.Draw(im)
    for ci,(c,cc) in enumerate(zip(cols,colc)): d.text((40+ci*(TILE+8), 12), c, font=fB, fill=cc)
    for ri,cd in enumerate(SEL):
        real = load(f"{D}/test/reals/{cd}_Rep1.TIF")
        geno = load(sorted(glob.glob(f"{GEN}/{cd}_*.png"))[0])
        genc = composite(geno)
        for ci,arr in enumerate([real, geno, genc]):
            im.paste(Image.fromarray(arr).resize((TILE,TILE)), (40+ci*(TILE+8), 44+ri*(TILE+8)))
    out = "/hpc/group/youlab/sa603/code/Data_augmentation/dish_composite_demo.png"; im.save(out); print("saved", out, SEL)
