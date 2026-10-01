"""Plot colony-feature downstream scarcity curve: mean R2 vs N_train conditions, per mode
(real / classical rotation / augmenter synth), error bars over seeds. Reads results.tsv."""
import sys, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TSV = sys.argv[1] if len(sys.argv) > 1 else "results.tsv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "fig_branching_downstream.png"

rows = list(csv.DictReader(open(TSV), delimiter="\t"))
# group[mode][N_train] = list of mean_r2 over seeds
group = defaultdict(lambda: defaultdict(list))
for r in rows:
    group[r["mode"]][r["N_train"]].append(float(r["mean_r2"]))

order = ["5", "10", "all"]
xs = {n: i for i, n in enumerate(order)}
colors = {"real": "#9aa0a6", "classical": "#fca5a5", "augmenter": "#86efac"}
labels = {"real": "real only", "classical": "real + rotation", "augmenter": "real + augmenter synth"}

plt.figure(figsize=(7, 5))
for mode in ["real", "classical", "augmenter"]:
    if mode not in group:
        continue
    ns = [n for n in order if n in group[mode]]
    means = [np.mean(group[mode][n]) for n in ns]
    stds = [np.std(group[mode][n]) for n in ns]
    plt.errorbar([xs[n] for n in ns], means, yerr=stds, marker="o", capsize=4,
                 color=colors.get(mode), label=labels.get(mode, mode), lw=2)
plt.xticks(range(len(order)), [f"N={n}" for n in order])
plt.xlabel("train conditions (data scarcity ->)"); plt.ylabel("mean R2 on held-out real (higher=better)")
plt.title("Colony-feature regression: does augmenter synth beat rotation under scarcity?")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUT, dpi=130); print("saved", OUT)

# also print a compact table
print(f"\n{'N_train':8s} " + " ".join(f"{m:>22s}" for m in ['real','classical','augmenter']))
for n in order:
    cells = []
    for m in ["real", "classical", "augmenter"]:
        v = group[m].get(n)
        cells.append(f"{np.mean(v):.3f}+-{np.std(v):.3f}" if v else "-")
    print(f"{n:8s} " + " ".join(f"{c:>22s}" for c in cells))
