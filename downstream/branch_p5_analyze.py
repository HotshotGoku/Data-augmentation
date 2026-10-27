"""Summarize branch_p5 results_p5.tsv. Per-arm means + paired (by N_train,seed) deltas vs rotation for:
  - shape_factor R2 (headline per-axis)
  - shape-5 aggregate  = mean R2 over {shape_factor,fractal_dim,solidity,eccentricity,edge_std}
  - branch-arch-4 aggregate = mean R2 over {n_tips,n_branch_points,branch_length,branch_density}
Usage: python branch_p5_analyze.py <results_p5.tsv>"""
import sys, csv, collections, statistics
from math import sqrt

SHAPE5 = ["r2_shape_factor", "r2_fractal_dim", "r2_solidity", "r2_eccentricity", "r2_edge_std"]
ARCH4 = ["r2_n_tips", "r2_n_branch_points", "r2_branch_length", "r2_branch_density"]
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))

def metrics(r):
    return {"shape_factor": float(r["r2_shape_factor"]),
            "shape5": statistics.mean(float(r[c]) for c in SHAPE5),
            "arch4": statistics.mean(float(r[c]) for c in ARCH4)}

# (mode, N_train) -> {seed: metrics}
by = collections.defaultdict(dict)
for r in rows:
    by[(r["mode"], r["N_train"])][int(r["seed"])] = metrics(r)

print("=== per-arm mean (over seeds) ===")
print(f"  {'arm':22} {'N':>4}  {'shape_factor':>12} {'shape5':>8} {'arch4':>8}  (n)")
for k in sorted(by):
    v = list(by[k].values())
    sf = statistics.mean(m["shape_factor"] for m in v)
    s5 = statistics.mean(m["shape5"] for m in v)
    a4 = statistics.mean(m["arch4"] for m in v)
    print(f"  {k[0]:22} {k[1]:>4}  {sf:12.4f} {s5:8.4f} {a4:8.4f}  ({len(v)})")

def paired(a, b, key):
    A, B = by.get(a, {}), by.get(b, {}); s = sorted(set(A) & set(B))
    if not s: return None
    d = [A[x][key] - B[x][key] for x in s]
    m = statistics.mean(d); sd = statistics.pstdev(d) or 1e-9
    return m, m / (sd / sqrt(len(d))), sum(1 for x in d if x > 0), len(s)

print("\n=== paired vs rotation (Δ>0 = beats rotation), per N_train ===")
nts = sorted({k[1] for k in by})
arms = [m for m in {k[0] for k in by} if m != "rotation"]
for nt in nts:
    print(f"  --- N_train={nt} ---")
    for arm in sorted(arms):
        for key in ("shape_factor", "shape5", "arch4"):
            res = paired((arm, nt), ("rotation", nt), key)
            if res:
                print(f"    {arm:22} {key:12} Δ={res[0]:+.4f} t={res[1]:+.2f} ({res[2]}/{res[3]} won)")
