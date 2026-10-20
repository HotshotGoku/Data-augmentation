"""Summary figures for the Kinshuk update deck (dark-theme, transparent bg to sit on dark slides)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
OUT = "/hpc/group/youlab/sa603/code/Data_augmentation/model_results"
BLUE, CORAL, GREEN, GRAY, TXT = "#7DD3FC", "#FCA5A5", "#86EFAC", "#9AA0A6", "#E9EAEC"
plt.rcParams.update({"text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT,
                     "ytick.color": TXT, "axes.edgecolor": "#2A2D34", "font.size": 12,
                     "font.family": "DejaVu Sans"})

def style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.18)

# 1) fine-tune lever across 3 domains
doms, base, ft = ["2-species", "multiplexed", "Rattray"], [2.54, 2.96, 2.18], [1.16, 1.52, 1.69]
x, w = np.arange(len(doms)), 0.36
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(x - w/2, base, w, label="base (zero-shot)", color=GRAY)
ax.bar(x + w/2, ft, w, label="fine-tuned", color=BLUE)
ax.axhline(1.0, ls="--", color=GREEN, lw=1.6, label="ideal (= real variability)")
for i, (b, f) in enumerate(zip(base, ft)):
    ax.text(i - w/2, b + 0.04, f"{b:.2f}×", ha="center", color=GRAY, fontsize=9)
    ax.text(i + w/2, f + 0.04, f"{f:.2f}×", ha="center", color=BLUE, fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(doms); ax.set_ylabel("realism ratio  (× real-vs-real; 1.0 = ideal)")
ax.set_ylim(0, 3.3); ax.legend(frameon=False, fontsize=9, loc="upper right"); style(ax); fig.tight_layout()
fig.savefig(f"{OUT}/fig_finetune_lever.png", dpi=150, transparent=True); plt.close(fig)

# 2) generalist per-domain
doms = ["kl2\n(2sp)", "ff\n(branch)", "selx\n(branch)", "mplex", "nlev\n(evo)", "pakp\n(Pa+Kp)"]
base, gen = [2.88, 2.71, 2.72, 2.96, 2.20, 3.16], [1.76, 2.21, 2.30, 2.30, 2.33, 3.26]
x, w = np.arange(len(doms)), 0.36
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.bar(x - w/2, base, w, label="base", color=GRAY)
ax.bar(x + w/2, gen, w, label="generalist (ONE model)", color=CORAL)
ax.axhline(1.0, ls="--", color=GREEN, lw=1.4)
ax.set_xticks(x); ax.set_xticklabels(doms, fontsize=9); ax.set_ylabel("realism ratio (1.0 = ideal)")
ax.set_ylim(0, 3.5); ax.legend(frameon=False, fontsize=9, loc="upper left"); style(ax); fig.tight_layout()
fig.savefig(f"{OUT}/fig_generalist.png", dpi=150, transparent=True); plt.close(fig)

# 3) VAE diagnostic — recon error vs the realism gap
doms, c4, c16 = ["multiplexed", "branching", "2-species"], [0.0198, 0.0343, 0.0261], [0.0099, 0.0154, 0.0078]
x, w = np.arange(len(doms)), 0.36
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.bar(x - w/2, c4, w, label="SD1.5 VAE (4-channel)", color=GRAY)
ax.bar(x + w/2, c16, w, label="16-channel VAE", color=BLUE)
ax.axhline(0.14, ls="--", color=CORAL, lw=2.0, label="realism gap to close (0.14)")
ax.set_xticks(x); ax.set_xticklabels(doms); ax.set_ylabel("VAE reconstruction LPIPS (lower = better)")
ax.set_ylim(0, 0.16)
ax.annotate("recon error ~7× BELOW the gap\n→ the VAE is not the bottleneck",
            xy=(0, 0.02), xytext=(0.55, 0.085), color=TXT, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=GRAY))
ax.legend(frameon=False, fontsize=9, loc="upper right"); style(ax); fig.tight_layout()
fig.savefig(f"{OUT}/fig_vae_diagnostic.png", dpi=150, transparent=True); plt.close(fig)
print("figs written:", OUT)
