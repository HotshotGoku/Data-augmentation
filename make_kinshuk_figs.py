"""Figures for the Kinshuk metrics-update slide (dark, transparent). Run on DCC (pytorch env)."""
import os, csv, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, cv2
OUT = "/hpc/group/youlab/sa603/code/Data_augmentation/model_results"
MPX_REAL = "/hpc/group/youlab/sa603/data/multiplexed_eval/reals_heldout"
PI = "/hpc/group/youlab/sa603/code/Data_augmentation/multiplexed_ft_eval_out/mplex_ft_epoch=2"
BLUE, CORAL, GREEN, GRAY, TXT = "#7DD3FC", "#FCA5A5", "#86EFAC", "#9AA0A6", "#E9EAEC"
plt.rcParams.update({"text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT,
                     "ytick.color": TXT, "axes.edgecolor": "#2A2D34", "font.size": 12, "font.family": "DejaVu Sans"})

def style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.18)

# 1) CMMD multiplexed (distribution metric; lower=better)
labs, vals = ["base", "2sp-ft", "mplex-ft"], [17.86, 9.70, 7.97]
fig, ax = plt.subplots(figsize=(5.6, 4.0))
b = ax.bar(labs, vals, color=[GRAY, BLUE, GREEN])
for i, v in enumerate(vals):
    ax.text(i, v + 0.3, f"{v:.1f}", ha="center", color=TXT, fontsize=11, fontweight="bold")
ax.set_ylabel("CMMD  (lower = better)"); ax.set_ylim(0, 20); style(ax); fig.tight_layout()
fig.savefig(f"{OUT}/fig_cmmd_mplex.png", dpi=150, transparent=True); plt.close(fig)

# 2) per-sample individual metrics (base vs mplex-ft)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 4.0))
a1.bar(["base", "mplex-ft"], [0.204, 0.500], color=[GRAY, GREEN])
a1.set_title("SSIM  (higher = better)", color=TXT, fontsize=11); a1.set_ylim(0, 0.6)
a2.bar(["base", "mplex-ft"], [0.862, 0.454], color=[GRAY, GREEN])
a2.set_title("LPIPS-VGG  (lower = better)", color=TXT, fontsize=11); a2.set_ylim(0, 1.0)
for a, vs in ((a1, [0.204, 0.500]), (a2, [0.862, 0.454])):
    for i, v in enumerate(vs):
        a.text(i, v + 0.01, f"{v:.2f}", ha="center", color=TXT, fontsize=10, fontweight="bold")
    style(a)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_persample.png", dpi=150, transparent=True); plt.close(fig)

# 3) generalist CMMD across 6 domains (base vs best generalist)
dom = ["pakp", "kl2", "ff", "nlev", "selx", "mplex"]
base = [44.86, 24.62, 49.06, 39.89, 47.28, 17.86]
gen = [11.20, 7.90, 18.70, 14.12, 17.45, 8.48]
x, w = np.arange(len(dom)), 0.38
fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.bar(x - w/2, base, w, label="base", color=GRAY)
ax.bar(x + w/2, gen, w, label="generalist (1 model)", color=CORAL)
ax.set_xticks(x); ax.set_xticklabels(dom); ax.set_ylabel("CMMD (lower = better)")
ax.legend(frameon=False, fontsize=10, loc="upper right"); style(ax); fig.tight_layout()
fig.savefig(f"{OUT}/fig_cmmd_generalist.png", dpi=150, transparent=True); plt.close(fig)

# 4) generated-vs-nearest-real montage from per_image.csv (biologist-facing; 3 best matches, distinct conditions)
rows = list(csv.DictReader(open(f"{PI}/per_image.csv")))
rows = [r for r in rows if r["nn_lpips_vgg"] not in ("", "nan")]
rows.sort(key=lambda r: float(r["nn_lpips_vgg"]))
picks, seen = [], set()
for r in rows:
    if r["prefix"] not in seen:
        picks.append(r); seen.add(r["prefix"])
    if len(picks) == 3:
        break
S, pad, top, left = 220, 8, 34, 118
W = left + 3 * (S + pad) + pad
H = top + 2 * (S + pad) + 40
canvas = np.full((H, W, 3), 20, np.uint8)
cv2.putText(canvas, "generated", (10, top + S // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
cv2.putText(canvas, "nearest real", (10, top + S + pad + S // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
for j, r in enumerate(picks):
    x0 = left + j * (S + pad) + pad
    g = cv2.resize(cv2.imread(r["gen_path"], cv2.IMREAD_COLOR), (S, S))
    rp = os.path.join(MPX_REAL, r["nearest_real"])
    real = cv2.resize(cv2.imread(rp, cv2.IMREAD_COLOR), (S, S))
    canvas[top:top + S, x0:x0 + S] = g
    canvas[top + S + pad:top + 2 * S + pad, x0:x0 + S] = real
    cv2.putText(canvas, f"{r['prefix']}  SSIM {float(r['nn_ssim']):.2f}  LPIPS {float(r['nn_lpips_vgg']):.2f}",
                (x0, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 200), 1)
cv2.imwrite(f"{OUT}/fig_nn_montage.png", canvas)
print("figs written:", OUT)
