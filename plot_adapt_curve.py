"""Adaptation data-efficiency curve (Rattray): realism vs #training pairs, base vs generalist, error bars."""
import os, csv, collections
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
T = os.environ.get("ADAPT_TSV", "/hpc/group/youlab/sa603/code/Data_augmentation/adapt_curve_out/curve_rattray.tsv")
OUT = os.environ.get("ADAPT_OUT", "/hpc/group/youlab/sa603/code/Data_augmentation/model_results/fig_adapt_curve_rattray.png")
BLUE, CORAL, GREEN, GRAY, TXT = "#7DD3FC", "#FCA5A5", "#86EFAC", "#9AA0A6", "#E9EAEC"
plt.rcParams.update({"text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT,
                     "ytick.color": TXT, "axes.edgecolor": "#2A2D34", "font.size": 12, "font.family": "DejaVu Sans"})
rows = list(csv.DictReader(open(T), delimiter="\t"))
agg = collections.defaultdict(lambda: collections.defaultdict(list))
floor = 0.32
for r in rows:
    agg[r["init"]][int(r["n_pairs"])].append(float(r["realism_sib"])); floor = float(r["base_sib"])
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for init, col in [("base", BLUE), ("generalist", CORAL)]:
    Ns = sorted(agg[init]); m = [np.mean(agg[init][n]) for n in Ns]; s = [np.std(agg[init][n]) for n in Ns]
    ax.errorbar(Ns, m, yerr=s, marker="o", color=col, lw=2.2, capsize=4, label=init)
ax.axhline(floor, ls="--", color=GREEN, lw=1.6, label=f"real-vs-real floor ({floor:.2f}) = ideal")
ax.set_xscale("log")
ax.set_xticks([10, 25, 50, 100, 250, 720]); ax.set_xticklabels(["10", "25", "50", "100", "250", "720\n(all)"])
ax.set_xlabel("# training pairs used to adapt (fine-tune)")
ax.set_ylabel("realism vs real replicates (LPIPS, lower = better)")
ax.set_ylim(0.3, 0.82)
ax.set_facecolor("none")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(alpha=0.18)
ax.legend(frameon=False, fontsize=11)
fig.tight_layout(); fig.savefig(OUT, dpi=150, transparent=True)
print("wrote", OUT)
