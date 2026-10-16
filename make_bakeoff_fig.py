"""Bake-off figure: no backbone's augmenter beats rotation on the new datasets.
Grouped bars, Δ vs rotation (15-seed), per backbone per dataset (all < 0)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
data = {"Rattray — strain ID": {"SD1.5 (best)": -0.108, "cond. DDPM": -0.196, "Flux": -0.258},
        "KL — mixing ratio":   {"SD1.5 (best)": -0.001, "cond. DDPM": -0.058, "Flux": -0.104}}
backbones = ["SD1.5 (best)", "cond. DDPM", "Flux"]
colors = {"SD1.5 (best)": "#86efac", "cond. DDPM": "#fca5a5", "Flux": "#7dd3fc"}
datasets = list(data); x = np.arange(len(datasets)); w = 0.26
fig, ax = plt.subplots(figsize=(8.2, 4.6)); fig.patch.set_facecolor("white")
for i, bb in enumerate(backbones):
    vals = [data[d][bb] for d in datasets]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=bb, color=colors[bb], edgecolor="#222")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.006, f"{v:+.3f}", ha="center", va="top", fontsize=8.5, fontweight="bold")
ax.axhline(0, color="#222", lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylabel("downstream Δ  (augmenter − rotation),  15 seeds", fontsize=10.5)
ax.set_title("No backbone's augmenter beats rotation on the new datasets\n(less-faithful backbones hurt more — rotation wins throughout)",
             fontsize=12.5, fontweight="bold")
ax.set_ylim(-0.30, 0.03); ax.legend(fontsize=9.5, loc="lower left", framealpha=0.9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(0.98, 0.04, "0 = ties rotation", transform=ax.transAxes, ha="right", fontsize=8.5, color="#555")
plt.tight_layout()
out = "/Users/sarthak/Downloads/Work/You Lab Research/backbone_viewer/bakeoff_result.png"
plt.savefig(out, dpi=160, bbox_inches="tight"); print("saved", out)
