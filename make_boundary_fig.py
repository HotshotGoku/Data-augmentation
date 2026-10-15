"""The cross-dataset boundary figure: generative augmentation vs rotation downstream, per dataset/axis.
Best augmenter config per dataset, paired delta vs rotation. Shows the axis-specific boundary."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, delta_vs_rotation, faithful?, note)  — best augmenter config per dataset, 15-seed
rows = [
    ("Branching — shape_factor",      +0.052, "win",  "dropout, t=2.14"),
    ("Multiplexed — aTc decode",      +0.045, "win",  "t=6.8, 14/15"),
    ("KL 2-species — mixing ratio",   -0.001, "tie",  "t=-0.06 (dead tie)"),
    ("Multiplexed — IPTG decode",     -0.022, "lose", "t=-2.2"),
    ("Rattray — strain ID",           -0.108, "lose", "t=-10.1, 0/15"),
]
rows.sort(key=lambda r: r[1])
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
cmap = {"win": "#2e7d32", "tie": "#9e9e9e", "lose": "#c62828"}
colors = [cmap[r[2]] for r in rows]

fig, ax = plt.subplots(figsize=(10, 4.6))
fig.patch.set_facecolor("white")
y = range(len(rows))
ax.barh(list(y), vals, color=colors, edgecolor="#222", height=0.62)
ax.axvline(0, color="#222", lw=1)
for i, r in enumerate(rows):
    off = 0.004 if r[1] >= 0 else -0.004
    ax.text(r[1] + off, i, f"{r[1]:+.3f}", va="center",
            ha="left" if r[1] >= 0 else "right", fontsize=10, fontweight="bold", color=cmap[r[2]])
    ax.text(0.153, i, r[3], va="center", ha="left", fontsize=8.5, color="#555", transform=ax.get_yaxis_transform())
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=10.5)
ax.set_xlabel("downstream metric Δ  (augmenter − rotation),  paired, 15 seeds", fontsize=10.5)
ax.set_xlim(-0.14, 0.14)
ax.set_title("Generative augmentation beats rotation only on faithful axes — not in general",
             fontsize=13, fontweight="bold", pad=12)
# region labels
ax.text(0.088, 3.35, "WINS\n(synth tracks the label)", fontsize=9,
        color="#2e7d32", ha="center", fontweight="bold")
ax.text(-0.052, 1.75, "LOSES\n(synth distorts the label;\nrotation preserves it exactly)", fontsize=9,
        color="#c62828", ha="center", va="center", fontweight="bold")
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout()
out = "/Users/sarthak/Downloads/Work/You Lab Research/backbone_viewer/boundary_result.png"
plt.savefig(out, dpi=160, bbox_inches="tight"); print("saved", out)
