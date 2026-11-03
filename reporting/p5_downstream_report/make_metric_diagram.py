"""Metric-explainer diagram for the Kinshuk report: visually defines shape_factor (round vs branchy),
branch-architecture (real colony: skeleton + tips at branch ends + junctions at splits), and the mux
aTc/IPTG color dials. Improved detection: iterative spur-prune, junction-cluster dedup, tips near edge. Run on DCC."""
import os, csv, glob
import numpy as np, cv2
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label as sklabel, regionprops
from scipy.ndimage import distance_transform_edt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "downstream")
from colony_features import load_gray, segment_colony   # same segmentation the labeler uses

D = "/hpc/group/youlab/sa603/data/clean"
CSV = "downstream/outputs/branching_downstream_out/features_branching_clean.csv"
OUT = "reporting/p5_downstream_report/figs"; os.makedirs(OUT, exist_ok=True)
def load(fp): return cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
def find_branch(name):
    for sub in ("canon", "test/sources", "test/reals"):
        p = f"{D}/branching_256/{sub}/{name}"
        if os.path.exists(p): return p
    g = glob.glob(f"{D}/branching_256/**/{name}", recursive=True); return g[0] if g else None

K = np.ones((3, 3), np.float32); K[1, 1] = 0
def _nb(s): return cv2.filter2D(s.astype(np.float32), -1, K) * s
def skel_points(fp):
    """Iterative spur-prune, dedup junction clusters, keep only tips near the colony edge (real branch ends)."""
    mask = segment_colony(load_gray(fp))
    if mask is None: return None, [], []
    sk = skeletonize(mask)
    for _ in range(4):
        nb = _nb(sk); kept = remove_small_objects(sklabel((sk > 0) & (nb < 3)), min_size=9) > 0
        new = skeletonize(kept | ((sk > 0) & (nb >= 3)))
        if new.sum() == sk.sum(): sk = new; break
        sk = new
    nb = _nb(sk); dist = distance_transform_edt(mask)
    junc = [r.centroid for r in regionprops(sklabel((sk > 0) & (nb >= 3)))]
    tips = [(y, x) for (y, x) in np.argwhere((sk > 0) & (nb == 1)) if dist[y, x] < 6]
    return sk, tips, junc

rows = [r for r in csv.DictReader(open(CSV)) if r["name"].endswith(".TIF") and find_branch(r["name"])]
rows.sort(key=lambda r: float(r["shape_factor"]))
branchy, round_ = rows[2], rows[-3]
mods = sorted([r for r in rows if 5 <= float(r["n_branch_points"]) <= 10], key=lambda r: float(r["shape_factor"]))
mod = mods[len(mods) // 2]   # a moderately-branched colony -> clean, accurate annotation

fig, ax = plt.subplots(1, 4, figsize=(15, 4.3))
ax[0].imshow(load(find_branch(round_["name"]))); ax[0].set_title("ROUND colony\nshape_factor is HIGH", fontsize=10)
ax[1].imshow(load(find_branch(branchy["name"]))); ax[1].set_title("BRANCHY colony\nshape_factor is LOW", fontsize=10)
bfp = find_branch(mod["name"]); ax[2].imshow(load(bfp))
sk, tips, junc = skel_points(bfp)
if sk is not None:
    ys, xs = np.where(sk > 0); ax[2].scatter(xs, ys, s=0.5, c="yellow", alpha=0.5, zorder=2)
    if tips: ax[2].scatter([p[1] for p in tips], [p[0] for p in tips], s=55, c="red", edgecolors="k", linewidths=0.4, zorder=3, label=f"tips = {len(tips)}")
    if junc: ax[2].scatter([p[1] for p in junc], [p[0] for p in junc], s=55, c="deepskyblue", edgecolors="k", linewidths=0.4, zorder=3, label=f"junctions = {len(junc)}")
    ax[2].legend(fontsize=8, loc="lower right")
ax[2].set_title("BRANCH-ARCHITECTURE\nred = tips (branch ends), blue = junctions (splits)", fontsize=9)
mux = sorted(glob.glob(f"{D}/multiplexed_eval/reals/*.TIF"))
ax[3].imshow(load(mux[len(mux) // 2])); ax[3].set_title("MUX pattern\naTc = red petals, IPTG = green fill", fontsize=10)
for a in ax: a.axis("off")
fig.text(0.5, 0.02,
         "shape_factor = round vs branchy   |   branch-architecture = number of tips & junctions in the skeleton   |   "
         "aTc / IPTG = the two chemical dials read from the mux colors",
         ha="center", fontsize=9)
fig.suptitle("What the measurements mean (plain-language guide to the terms in the graphs)", fontsize=13)
plt.tight_layout(rect=[0, 0.05, 1, 0.96]); plt.savefig(f"{OUT}/metrics_explainer.png", dpi=150, bbox_inches="tight"); plt.close()
print("WROTE | round", round_["name"], "| branchy", branchy["name"], "| annotated(moderate)", mod["name"],
      "tips", len(tips), "junc", len(junc))
