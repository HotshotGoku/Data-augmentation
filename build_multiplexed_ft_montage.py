"""Montage: does the multiplexed fine-tune actually reproduce the pattern (not just win the metric)?
Columns: source | base | 2sp-ft | mplex-ft(ep2) | real sibling. Rows: conditions. Run on DCC."""
import os, glob, cv2, numpy as np

MPX = "/hpc/group/youlab/sa603/data/multiplexed_eval"
EVAL = "/hpc/group/youlab/sa603/code/Data_augmentation/multiplexed_ft_eval_out"
OUT = os.environ.get("MONT_OUT", "/hpc/group/youlab/sa603/code/Data_augmentation/model_results/multiplexed_ft_comparison.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
S, PAD, LAB_L, LAB_T = 256, 6, 150, 34
COLS = [("source", MPX + "/sources", "{c}_src.TIF"),
        ("base", EVAL + "/base/gen", "{c}*"),
        ("2sp-ft", EVAL + "/ft_2species/gen", "{c}*"),
        ("mplex-ft (ep2)", EVAL + "/mplex_ft_epoch=2/gen", "{c}*"),
        ("real sibling", MPX + "/reals_heldout", "{c}_Rep*")]
conds = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(MPX + "/sources/*.TIF")})[:int(os.environ.get("MONT_N", "5"))]

def load(base, pat, c):
    fs = [f for f in sorted(glob.glob(os.path.join(base, pat.format(c=c))))
          if f.lower().endswith((".png", ".tif", ".jpg", ".jpeg"))]
    if not fs:
        return np.full((S, S, 3), 40, np.uint8)
    return cv2.resize(cv2.imread(fs[0], cv2.IMREAD_COLOR), (S, S))

W = LAB_L + len(COLS) * (S + PAD) + PAD
H = LAB_T + len(conds) * (S + PAD) + PAD
canvas = np.full((H, W, 3), 20, np.uint8)
for j, (name, _, _) in enumerate(COLS):
    x = LAB_L + j * (S + PAD) + PAD
    cv2.putText(canvas, name, (x + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
for i, c in enumerate(conds):
    y = LAB_T + i * (S + PAD) + PAD
    cv2.putText(canvas, c, (8, y + S // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    for j, (_, base, pat) in enumerate(COLS):
        x = LAB_L + j * (S + PAD) + PAD
        canvas[y:y + S, x:x + S] = load(base, pat, c)
cv2.imwrite(OUT, canvas)
print("wrote", OUT, canvas.shape)
