"""Money plot for the downstream utility experiment: joint decode accuracy vs K, 3 arms + ceiling.
Dark-theme, transparent bg to match the deck; no in-plot title (the slide provides it)."""
import csv, collections
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/hpc/group/youlab/sa603/code/Data_augmentation/downstream_multiplexed_out/results.tsv"
OUT = "/hpc/group/youlab/sa603/code/Data_augmentation/model_results/downstream_utility.png"
BLUE, CORAL, GREEN, GRAY, TXT = "#7DD3FC", "#FCA5A5", "#86EFAC", "#9AA0A6", "#E9EAEC"
plt.rcParams.update({"text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT,
                     "ytick.color": TXT, "axes.edgecolor": "#2A2D34", "font.size": 12,
                     "font.family": "DejaVu Sans"})
rows = list(csv.DictReader(open(R), delimiter="\t"))
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["mode"], r["K"])].append(float(r["joint_acc"]))

def curve(m, ks):
    x, y, e = [], [], []
    for k in ks:
        v = agg.get((m, str(k)))
        if v:
            x.append(k); y.append(np.mean(v)); e.append(np.std(v))
    return x, y, e

Ks = [1, 2, 3]
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for m, c in [("real", GRAY), ("classical", BLUE), ("augmenter", CORAL)]:
    x, y, e = curve(m, Ks)
    ax.errorbar(x, y, yerr=e, marker="o", capsize=4, color=c, lw=2.4, label=m)
ceil = np.mean(agg[("real", "all")])
ax.axhline(ceil, ls="--", color=GREEN, lw=1.8, label=f"real all-reps ceiling ({ceil:.2f})")
ax.axhline(1 / 70, ls=":", color="#6B7076", label="chance (1/70)")
ax.set_xticks(Ks); ax.set_xlabel("real replicates per condition (K)")
ax.set_ylabel("joint Row+Col decode accuracy (held-out real test)")
ax.set_facecolor("none")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(alpha=0.18)
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=150, transparent=True)
print("wrote", OUT)
