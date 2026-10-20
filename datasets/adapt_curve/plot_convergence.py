"""WS-F convergence panel (Rattray): realism vs epoch for a SMALL N and the FULL set, base init.
Both plateau within ~1 epoch -> adaptation is epoch-driven, so the N-gap is a DATA effect (more data
= better replicates), NOT a training-amount effect. Rebuts 'small N just trained fewer steps'."""
import os, csv, collections
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
T = os.environ.get("CONV_TSV", "/hpc/group/youlab/sa603/code/Data_augmentation/adapt_curve_out/convergence_rattray.tsv")
OUT = os.environ.get("CONV_OUT", "/hpc/group/youlab/sa603/code/Data_augmentation/model_results/fig_convergence_rattray.png")
BLUE, CORAL, GREEN, TXT = "#7DD3FC", "#FCA5A5", "#86EFAC", "#E9EAEC"
plt.rcParams.update({"text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT,
                     "ytick.color": TXT, "axes.edgecolor": "#2A2D34", "font.size": 12, "font.family": "DejaVu Sans"})
rows = list(csv.DictReader(open(T), delimiter="\t"))
by = collections.defaultdict(dict); floor = 0.32
for r in rows:
    by[r["n_pairs"]][int(r["epoch"])] = float(r["realism_sib"]); floor = float(r["base_sib"])
series = sorted(by, key=lambda k: int(k))            # e.g. ["25", "720"]
colors = {series[0]: CORAL, series[-1]: BLUE}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for npairs in series:
    eps = sorted(by[npairs]); ys = [by[npairs][e] for e in eps]
    lab = f"N={npairs} pairs" + (" (all)" if npairs == series[-1] else "")
    ax.plot([e + 1 for e in eps], ys, marker="o", lw=2.2, color=colors.get(npairs, BLUE), label=lab)
ax.axhline(floor, ls="--", color=GREEN, lw=1.6, label=f"real-vs-real floor ({floor:.2f}) = ideal")
ax.set_xlabel("epoch of fine-tuning")
ax.set_ylabel("realism vs real replicates (LPIPS, lower = better)")
ax.set_ylim(0.3, 0.82)
ax.set_facecolor("none")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(alpha=0.18)
ax.legend(frameon=False, fontsize=11, title="adapt in ~1 epoch; gap = data, not training")
fig.tight_layout(); fig.savefig(OUT, dpi=150, transparent=True)
print("wrote", OUT)
