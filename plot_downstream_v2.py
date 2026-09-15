"""Downstream v2: regression R2 (aTc, IPTG) vs K, 4 arms, error bars (5 seeds). Dark/transparent."""
import csv, collections
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/hpc/group/youlab/sa603/code/Data_augmentation/downstream_multiplexed_out/results_v2.tsv"
OUT = "/hpc/group/youlab/sa603/code/Data_augmentation/model_results/fig_downstream_v2.png"
GRAY, BLUE, CORAL, PURP, GREEN, TXT = "#9AA0A6", "#7DD3FC", "#FCA5A5", "#C4B5FD", "#86EFAC", "#E9EAEC"
plt.rcParams.update({"text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT,
                     "ytick.color": TXT, "axes.edgecolor": "#2A2D34", "font.size": 12, "font.family": "DejaVu Sans"})
rows = list(csv.DictReader(open(R), delimiter="\t"))
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    agg[(r["mode"], r["K"])]["atc"].append(float(r["atc_r2"]))
    agg[(r["mode"], r["K"])]["iptg"].append(float(r["iptg_r2"]))
arms = [("real", GRAY), ("classical", BLUE), ("aug_mplexft", CORAL), ("aug_general", PURP)]
Ks = ["1", "2", "3"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
for ax, key, title in ((axes[0], "atc", "aTc readout (R², higher = better)"),
                       (axes[1], "iptg", "IPTG readout (R², higher = better)")):
    for arm, col in arms:
        m = [np.mean(agg[(arm, k)][key]) for k in Ks]
        s = [np.std(agg[(arm, k)][key]) for k in Ks]
        ax.errorbar([1, 2, 3], m, yerr=s, marker="o", color=col, lw=2, capsize=4, label=arm)
    ra = np.mean(agg[("real", "all")][key])
    ax.axhline(ra, ls="--", color=GREEN, lw=1.5, label="real, ALL data")
    ax.set_title(title, color=TXT, fontsize=12); ax.set_xlabel("# real replicates / condition (K)")
    ax.set_xticks([1, 2, 3]); ax.set_ylabel("R²"); ax.set_ylim(-0.1, 0.8)
    ax.set_facecolor("none")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.18)
axes[1].legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(OUT, dpi=150, transparent=True)
print("wrote", OUT)
