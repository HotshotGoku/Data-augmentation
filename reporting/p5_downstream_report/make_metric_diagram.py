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
import sys; sys.path.insert(0, "downstream")
from colony_features import load_gray, segment_colony   # use the SAME clean binary the labeler uses

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
    if not HAVE_SK: return None, None, None
    from skimage.morphology import remove_small_objects
    from skimage.measure import label as sklabel, regionprops
    mask = segment_colony(load_gray(fp))   # largest colony only, holes filled - no noise specks
    if mask is None: return None, None, None
    sk = skeletonize(mask)
    kk = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)
    nb0 = cv2.filter2D(sk.astype(np.float32), -1, kk) * sk
    kept = remove_small_objects(sklabel(sk & (nb0 < 3)), min_size=6) > 0   # prune <6px spurs
    sk = skeletonize(kept | (sk & (nb0 >= 3)))
    nb = cv2.filter2D(sk.astype(np.float32), -1, kk) * sk
    tips = np.array([r.centroid for r in regionprops(sklabel(nb == 1))]) if (nb == 1).any() else np.empty((0, 2))
    junc = np.array([r.centroid for r in regionprops(sklabel(nb >= 3))]) if (nb >= 3).any() else np.empty((0, 2))
    return sk, tips, junc   # one centroid per tip/junction CLUSTER

fig, ax = plt.subplots(1, 4, figsize=(14, 4.2))
ax[0].imshow(load(find_branch(round_["name"]))); ax[0].set_title("ROUND colony\nshape_factor is HIGH", fontsize=10)
ax[1].imshow(load(find_branch(branchy["name"]))); ax[1].set_title("BRANCHY colony\nshape_factor is LOW", fontsize=10)
# panel 3: CLEAN SCHEMATIC (idealized tree) - defines tip/junction unambiguously (not auto-detected)
a2 = ax[2]; a2.set_xlim(-0.05, 1.05); a2.set_ylim(-0.1, 1.12); a2.set_aspect("equal")
C = (0.5, 0.5)
J = [(0.20, 0.74), (0.80, 0.74), (0.24, 0.26), (0.76, 0.26)]
T = [(0.05, 0.95), (0.34, 0.99), (0.66, 0.99), (0.95, 0.95),
     (0.05, 0.05), (0.34, 0.01), (0.66, 0.01), (0.95, 0.05)]
links = [(C, J[0]), (C, J[1]), (C, J[2]), (C, J[3]),
         (J[0], T[0]), (J[0], T[1]), (J[1], T[2]), (J[1], T[3]),
         (J[2], T[4]), (J[2], T[5]), (J[3], T[6]), (J[3], T[7])]
for (x0, y0), (x1, y1) in links:
    a2.plot([x0, x1], [y0, y1], c="#999", lw=2.6, zorder=1, solid_capstyle="round")
jp = [C] + J
a2.scatter([p[0] for p in jp], [p[1] for p in jp], s=120, c="deepskyblue", edgecolors="k", linewidths=0.6, zorder=3)
a2.scatter([p[0] for p in T], [p[1] for p in T], s=120, c="red", edgecolors="k", linewidths=0.6, zorder=3)
a2.text(0.5, 0.59, "junction", fontsize=8, color="#06c", ha="center", va="bottom", zorder=4, fontweight="bold")
a2.text(0.34, 0.90, "tip", fontsize=8, color="#c00", ha="center", va="top", zorder=4, fontweight="bold")
a2.set_title("BRANCH-ARCHITECTURE (schematic)\nred = tips (branch ends) = 8,  blue = junctions (splits) = 5", fontsize=8.5)
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
