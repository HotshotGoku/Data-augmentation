"""Metric-explainer diagram for the Kinshuk report: visually defines shape_factor (round vs branchy),
branch-architecture (skeleton tips + junctions), and the mux aTc/IPTG color dials. Run on DCC."""
import os, csv, glob
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from skimage.morphology import skeletonize
    HAVE_SK = True
except Exception:
    HAVE_SK = False

D = "/hpc/group/youlab/sa603/data/clean"
CSV = "downstream/outputs/branching_downstream_out/features_branching_clean.csv"
OUT = "reporting/p5_downstream_report/figs"; os.makedirs(OUT, exist_ok=True)
def load(fp): return cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
def find_branch(name):
    for sub in ("canon", "test/reals", "test/sources"):
        p = f"{D}/branching_256/{sub}/{name}"
        if os.path.exists(p): return p
    g = glob.glob(f"{D}/branching_256/**/{name}", recursive=True); return g[0] if g else None

rows = [r for r in csv.DictReader(open(CSV)) if r["name"].endswith(".TIF") and find_branch(r["name"])]
rows.sort(key=lambda r: float(r["shape_factor"]))
branchy, round_ = rows[2], rows[-3]   # low shape_factor = branchy, high = round (skip extremes)

def skel(fp):
    g = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    if not HAVE_SK: return None, None, None
    sk = skeletonize(g > 35)
    kk = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)
    nb = cv2.filter2D(sk.astype(np.float32), -1, kk)
    tips = np.argwhere(sk & (nb == 1)); junc = np.argwhere(sk & (nb >= 3))
    return sk, tips, junc

fig, ax = plt.subplots(1, 4, figsize=(14, 4.2))
ax[0].imshow(load(find_branch(round_["name"]))); ax[0].set_title("ROUND colony\nshape_factor is HIGH", fontsize=10)
ax[1].imshow(load(find_branch(branchy["name"]))); ax[1].set_title("BRANCHY colony\nshape_factor is LOW", fontsize=10)
bfp = find_branch(branchy["name"]); ax[2].imshow(load(bfp))
sk, tips, junc = skel(bfp)
if sk is not None:
    ys, xs = np.where(sk); ax[2].scatter(xs, ys, s=0.4, c="yellow", alpha=0.45)
    if len(tips): ax[2].scatter(tips[:, 1], tips[:, 0], s=20, c="red", edgecolors="k", linewidths=0.3, label=f"tips = {branchy['n_tips']}")
    if len(junc): ax[2].scatter(junc[:, 1], junc[:, 0], s=20, c="deepskyblue", edgecolors="k", linewidths=0.3, label=f"junctions = {branchy['n_branch_points']}")
    ax[2].legend(fontsize=7, loc="lower right")
ax[2].set_title("BRANCH-ARCHITECTURE\ncount the tips & junctions", fontsize=10)
mux = sorted(glob.glob(f"{D}/multiplexed_eval/reals/*.TIF"))
ax[3].imshow(load(mux[len(mux)//2])); ax[3].set_title("MUX pattern\naTc = red petals, IPTG = green fill", fontsize=10)
for a in ax: a.axis("off")
fig.text(0.5, 0.02,
         "shape_factor = round vs branchy   |   branch-architecture = number of tips & junctions in the skeleton   |   "
         "aTc / IPTG = the two chemical dials the model reads from the mux colors",
         ha="center", fontsize=9)
fig.suptitle("What the measurements mean (plain-language guide to the terms in the graphs)", fontsize=13)
plt.tight_layout(rect=[0, 0.05, 1, 0.96]); plt.savefig(f"{OUT}/metrics_explainer.png", dpi=150, bbox_inches="tight"); plt.close()
print("WROTE metrics_explainer.png | round", round_["name"], round(float(round_["shape_factor"]), 2),
      "| branchy", branchy["name"], round(float(branchy["shape_factor"]), 2), "| skimage", HAVE_SK)
