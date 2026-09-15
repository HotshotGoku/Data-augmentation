"""Generalist CMMD (336px, paper config) across 6 domains: base vs generalist. Dark/transparent."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
OUT = "/hpc/group/youlab/sa603/code/Data_augmentation/model_results/fig_cmmd_generalist_336.png"
GRAY, CORAL, GREEN, TXT = "#9AA0A6", "#FCA5A5", "#86EFAC", "#E9EAEC"
plt.rcParams.update({"text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT,
                     "ytick.color": TXT, "axes.edgecolor": "#2A2D34", "font.size": 12, "font.family": "DejaVu Sans"})
dom = ["pakp", "kl2", "ff", "nlev", "selx", "mplex"]
base = [32.93, 29.73, 39.30, 32.79, 36.72, 19.73]
gen = [9.78, 13.52, 10.53, 7.91, 10.59, 12.49]
x, w = np.arange(len(dom)), 0.38
fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.bar(x - w/2, base, w, label="base", color=GRAY)
ax.bar(x + w/2, gen, w, label="generalist (1 model)", color=CORAL)
ax.set_xticks(x); ax.set_xticklabels(dom); ax.set_ylabel("CMMD (lower = better)")
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.set_facecolor("none")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.18)
fig.tight_layout(); fig.savefig(OUT, dpi=150, transparent=True)
print("wrote", OUT)
