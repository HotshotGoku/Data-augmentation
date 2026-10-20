"""Two deck charts (dark theme): (1) Flux escalation ladder vs in-domain SD1.5 on realism+CMMD;
(2) 15-seed downstream mean-R2 with error bars (rotation strong, all within subset noise)."""
import os
os.environ["MPLCONFIGDIR"] = "/private/tmp/claude/mpl"
import csv, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, FG, GRID = "#0B0C0F", "#E9EAEC", "#2A2D33"
GRN, CORAL, AMBER, BLUE, GRAY = "#86EFAC", "#FCA5A5", "#FBBF24", "#7DD3FC", "#9AA0A6"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": GRID, "font.size": 12})

# ---- Chart 1: escalation ladder ----
labels = ["in-domain\nSD1.5", "Flux pilot\n(fp8/CN2)", "Flux scaled\n(fp8/CN4)", "Flux bf16\n(CN6)"]
realism = [0.324, 0.524, 0.499, 0.503]
cmmd = [3.38, 8.78, 5.53, 4.84]
cols = [GRN, CORAL, AMBER, BLUE]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, vals, ttl in [(a1, realism, "Realism vs real siblings  (lower = better)"),
                      (a2, cmmd, "CMMD  (lower = better)")]:
    b = ax.bar(range(4), vals, color=cols, width=0.62)
    ax.set_title(ttl, fontsize=12, color=FG, pad=10)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbisebelow if False else None
    for r, v in zip(b, vals): ax.text(r.get_x()+r.get_width()/2, v, f"{v:.3f}" if max(vals)<1 else f"{v:.2f}",
        ha="center", va="bottom", fontsize=10, color=FG)
a1.axhline(0.253, color=GRAY, ls="--", lw=1); a1.text(3.4, 0.263, "real-vs-real floor 0.25", ha="right", fontsize=8, color=GRAY)
a1.set_ylim(0, 0.60)
fig.suptitle("Flux never beats in-domain SD1.5 on realism — CMMD improves with every lever but plateaus above SD1.5",
             fontsize=12.5, color=FG, y=1.02)
fig.tight_layout(); fig.savefig("/private/tmp/claude/chart_ladder.png", dpi=150, bbox_inches="tight")
print("saved chart_ladder.png")

# ---- Chart 2: 15-seed downstream ----
rows = []
for r in csv.DictReader(open("/private/tmp/claude/results_15seed.tsv"), delimiter="\t"):
    r["mean_r2"] = float(r["mean_r2"]); rows.append(r)
ARMS = [("real", "real only", GRAY), ("classical", "+rotation", GRN),
        ("augmenter_base", "+SD1.5 synth", BLUE), ("augmenter_flux", "+Flux synth", CORAL)]
def ms(mode, N):
    v = [x["mean_r2"] for x in rows if x["mode"] == mode and x["N_train"] == N]
    return st.mean(v), st.stdev(v)/len(v)**0.5  # SEM
fig, ax = plt.subplots(figsize=(8.8, 4.8))
x = np.arange(2); w = 0.2
for i, (m, lab, c) in enumerate(ARMS):
    means = [ms(m, "20")[0], ms(m, "60")[0]]; sems = [ms(m, "20")[1], ms(m, "60")[1]]
    ax.bar(x + (i-1.5)*w, means, w, yerr=sems, capsize=4, label=lab, color=c,
           error_kw=dict(ecolor=FG, lw=1.1))
ax.set_xticks(x); ax.set_xticklabels(["N=20 conditions", "N=60 conditions"])
ax.set_ylabel("downstream mean R²  (15 seeds, ±SEM)"); ax.set_ylim(0, 0.55)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=10, ncol=4, loc="upper left")
ax.set_title("Downstream: rotation ≈ SD1.5 ≈ Flux — no augmenter significantly beats plain rotation",
             fontsize=12, color=FG, pad=10)
fig.tight_layout(); fig.savefig("/private/tmp/claude/chart_downstream.png", dpi=150, bbox_inches="tight")
print("saved chart_downstream.png")
