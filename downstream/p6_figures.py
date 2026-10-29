"""P6 figures for the P5 downstream aug-vs-rotation study. Honest, axis-selective story.
Reads the results_p5 TSVs; writes PNGs to downstream/outputs/p6_figures/. Run on DCC (matplotlib)."""
import os, csv, statistics as st
from math import sqrt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "downstream/outputs/p6_figures"; os.makedirs(OUT, exist_ok=True)
BR = "downstream/outputs/branching_downstream_out"
MX = "downstream/outputs/downstream_multiplexed_out"
MP = "downstream/outputs/mux_periodicity_out"
def load(p): return list(csv.DictReader(open(p), delimiter="\t"))
def ci95(d): m = st.mean(d); s = st.pstdev(d)/sqrt(len(d)) if len(d) > 1 else 0; return m, 1.96*s
ARCH = ["r2_n_tips", "r2_n_branch_points", "r2_branch_length", "r2_branch_density"]
def arch4(r): return st.mean(float(r[c]) for c in ARCH)
def sf(r): return float(r["r2_shape_factor"])

brg = load(f"{BR}/results_p5_gray.tsv"); brr = load(f"{BR}/results_p5.tsv")
mdec = load(f"{MX}/results_p5.tsv"); mper = load(f"{MP}/results_p5.tsv")
kl = load("downstream/outputs/kl_downstream_p5.tsv"); rat = load("downstream/outputs/rattray_downstream_p5.tsv")

def pdelta(rows, scar_col, scar, armA, metric):   # paired aug-rotation per seed within a scarcity cell
    A = {r["seed"]: metric(r) for r in rows if r["mode"] == armA and r[scar_col] == scar}
    R = {r["seed"]: metric(r) for r in rows if r["mode"] == "rotation" and r[scar_col] == scar}
    return [A[s] - R[s] for s in sorted(set(A) & set(R))]
def ds_pdelta(rows, aug_tag):                     # kl/rattray: tag-keyed aug vs rotation
    A = {r["seed"]: float(r["metric"]) for r in rows if r["tag"] == aug_tag and r["mode"] == "augmenter"}
    R = {r["seed"]: float(r["metric"]) for r in rows if r["mode"] == "rotation"}
    return [A[s] - R[s] for s in sorted(set(A) & set(R))]
def cell(rows, arm, scar_col, scar, metric):      # arm mean+sem within a cell
    v = [metric(r) for r in rows if r["mode"] == arm and r[scar_col] == scar]
    return (st.mean(v), st.pstdev(v)/sqrt(len(v))) if v else (None, None)

# ---------- Fig 1: headline forest (aug - rotation, per task-axis; config/scarcity annotated) ----------
forest = [
    ("mux  aTc  (row_acc)  dropout K3",      pdelta(mdec, "K", "3", "augmenter_dropout", lambda r: float(r["row_acc"]))),
    ("branching  branch-arch  stratsm N20",  pdelta(brg, "N_train", "20", "augmenter_stratsm", arch4)),
    ("branching  branch-arch  dropout N20",  pdelta(brg, "N_train", "20", "augmenter_dropout", arch4)),
    ("branching  shape_factor  dropout N20", pdelta(brg, "N_train", "20", "augmenter_dropout", sf)),
    ("mux  IPTG  (col_acc)  dropout K3",     pdelta(mdec, "K", "3", "augmenter_dropout", lambda r: float(r["col_acc"]))),
    ("mux  periodicity  base K3",            pdelta(mper, "K", "3", "augmenter_base", lambda r: float(r["r2_perim_roughness"]))),
    ("rattray  strain  indomain",            ds_pdelta(rat, "rattray_indomain")),
    ("branching  shape_factor  clean N60",   pdelta(brg, "N_train", "60", "augmenter_clean", sf)),
    ("kl  ratio  indomain",                  ds_pdelta(kl, "kl_indomain")),
]
forest = [(lbl, *ci95(d)) for lbl, d in forest if d]
forest.sort(key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(9, 5.5)); ys = range(len(forest))
for y, (lbl, m, ci) in zip(ys, forest):
    col = "#2a8" if m - ci > 0 else ("#c33" if m + ci < 0 else "#999")
    ax.errorbar(m, y, xerr=ci, fmt="o", color=col, capsize=3, ms=7)
ax.axvline(0, color="k", lw=.8); ax.set_yticks(list(ys)); ax.set_yticklabels([f[0] for f in forest], fontsize=8)
ax.set_xlabel("augmenter − rotation  (paired Δ, 15 seeds, 95% CI)")
ax.set_title("P5 downstream: generative augmentation is AXIS-SELECTIVE\n(green=beats rotation, red=loses, gray=ns)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_headline_forest.png", dpi=150); plt.close()

# ---------- Fig 2: branching scarcity curves (grayscale = fair), shape_factor vs branch-arch ----------
arms = [("real", "#888"), ("real_dup", "#58a"), ("rotation", "#000"),
        ("augmenter_clean", "#e88"), ("augmenter_dropout", "#d22"), ("augmenter_stratsm", "#811")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric, ttl in [(a1, sf, "shape_factor  (headline)"), (a2, arch4, "branch-architecture (arch4)")]:
    for arm, c in arms:
        pts = [cell(brg, arm, "N_train", n, metric) for n in ("20", "60")]
        ax.errorbar([20, 60], [p[0] for p in pts], yerr=[p[1] for p in pts], marker="o", color=c,
                    label=arm.replace("augmenter_", "aug_"), lw=1.8)
    cy, _ = cell(brg, "real_ceiling", "N_train", "all", metric)
    if cy is not None: ax.axhline(cy, ls="--", color="g", alpha=.6, label="ceiling (all real)")
    ax.set_xlabel("N_train conditions"); ax.set_ylabel("test R²"); ax.set_title(ttl); ax.set_xticks([20, 60])
a1.legend(fontsize=7, loc="upper left")
fig.suptitle("Branching scarcity (grayscale inputs = tint-free): aug loses shape_factor, wins branch-arch")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_branching_scarcity.png", dpi=150); plt.close()

# ---------- Fig 3: mux decode aTc(row) vs IPTG(col) axis-selectivity ----------
marms = [("real", "#888"), ("real_dup", "#58a"), ("rotation", "#000"), ("augmenter_base", "#d22"), ("augmenter_dropout", "#811")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, ttl in [(a1, "row_acc", "aTc  (row_acc)  — WIN"), (a2, "col_acc", "IPTG  (col_acc)  — lose")]:
    for arm, c in marms:
        pts = [cell(mdec, arm, "K", k, lambda r: float(r[col])) for k in ("1", "2", "3")]
        ax.errorbar([1, 2, 3], [p[0] for p in pts], yerr=[p[1] for p in pts], marker="o", color=c,
                    label=arm.replace("augmenter_", "aug_"), lw=1.8)
    cy, _ = cell(mdec, "real_ceiling", "K", "all", lambda r: float(r[col]))
    if cy is not None: ax.axhline(cy, ls="--", color="g", alpha=.6, label="ceiling")
    ax.set_xlabel("K (real reps/condition)"); ax.set_ylabel("accuracy"); ax.set_title(ttl); ax.set_xticks([1, 2, 3])
a1.legend(fontsize=8)
fig.suptitle("Mux decode: augmenter beats rotation on aTc (row), loses on IPTG (col) — axis-selective")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_mux_axis_selectivity.png", dpi=150); plt.close()

# ---------- Fig 4: color-tint confound (RGB vs grayscale), aug-rotation shape_factor Δ ----------
cfgs = ["augmenter_clean", "augmenter_dropout", "augmenter_stratsm"]
fig, ax = plt.subplots(figsize=(9, 5)); labels = []; i = 0
for n in ("20", "60"):
    for cfg in cfgs:
        rgb = ci95(pdelta(brr, "N_train", n, cfg, sf)); gry = ci95(pdelta(brg, "N_train", n, cfg, sf))
        ax.bar(i-0.18, rgb[0], 0.36, yerr=rgb[1], color="#c93", label="RGB (tinted)" if i == 0 else "")
        ax.bar(i+0.18, gry[0], 0.36, yerr=gry[1], color="#48a", label="grayscale" if i == 0 else "")
        labels.append(f"{cfg.replace('augmenter_','')}\nN={n}"); i += 1
ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(i)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("augmenter − rotation  (shape_factor Δ)"); ax.legend()
ax.set_title("Color-tint confound: grayscaling recovers ~0.07–0.14 R²,\nbut augmenter still ≤ rotation on shape_factor (genuine, milder negative)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_tint_confound.png", dpi=150); plt.close()

# ---------- Fig 5: documented negatives (kl, rattray, periodicity) per-arm ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
def ds_arm(rows, tag, mode):
    v = [float(r["metric"]) for r in rows if r["tag"] == tag and r["mode"] == mode]; return st.mean(v) if v else 0
for ax, rows, ds, ttl, yl in [(axes[0], kl, "kl", "kl  ratio  (R²)", "R²"), (axes[1], rat, "rattray", "rattray  strain  (acc)", "accuracy")]:
    bars = [("real", ds_arm(rows, ds, "real")), ("real_dup", ds_arm(rows, ds, "real_dup")),
            ("rotation", ds_arm(rows, ds, "rotation")), ("aug_indomain", ds_arm(rows, f"{ds}_indomain", "augmenter")),
            ("aug_dropout", ds_arm(rows, f"{ds}_dropout", "augmenter")), ("ceiling", ds_arm(rows, f"{ds}_ceiling", "real"))]
    cols = ["#888", "#58a", "#000", "#d22", "#811", "#2a8"]
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=cols); ax.set_title(ttl); ax.set_ylabel(yl)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
pa = axes[2]
parms = [("real", "#888"), ("real_dup", "#58a"), ("rotation", "#000"), ("augmenter_base", "#d22"), ("augmenter_dropout", "#811"), ("real_ceiling", "#2a8")]
vals = [cell(mper, a, "K", "3" if a != "real_ceiling" else "all", lambda r: float(r["r2_perim_roughness"]))[0] for a, _ in parms]
pa.bar([a.replace("augmenter_", "aug_").replace("real_ceiling", "ceiling") for a, _ in parms], vals, color=[c for _, c in parms])
pa.set_title("mux periodicity  perim_roughness  (R², K=3)"); pa.set_ylabel("R²"); pa.tick_params(axis="x", rotation=30, labelsize=8)
fig.suptitle("Documented negatives: augmenter < rotation < ceiling (synth carry no faithful label on these axes)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_negatives.png", dpi=150); plt.close()

print("WROTE:", sorted(os.listdir(OUT)))
