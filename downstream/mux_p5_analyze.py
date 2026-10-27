"""Summarize mux P5 results_p5.tsv (auto-detects DECODE vs PERIODICITY by header).
  DECODE      -> row_acc(aTc, headline), col_acc(IPTG), joint_acc, atc_r2, iptg_r2
  PERIODICITY -> r2_perim_roughness (headline), r2_n_lobes, mean_r2
Per-arm means + paired (by K,seed) deltas vs rotation. Usage: python mux_p5_analyze.py <results_p5.tsv>"""
import sys, csv, collections, statistics
from math import sqrt

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
hdr = rows[0].keys() if rows else []
if "row_acc" in hdr:
    KEYS = ["row_acc", "col_acc", "joint_acc", "atc_r2", "iptg_r2"]; HEAD = "row_acc(aTc)"
elif "r2_perim_roughness" in hdr:
    KEYS = ["r2_perim_roughness", "r2_n_lobes", "mean_r2"]; HEAD = "r2_perim_roughness"
else:
    sys.exit(f"unrecognized TSV header: {list(hdr)}")

by = collections.defaultdict(dict)   # (mode, K) -> {seed: {key: val}}
for r in rows:
    by[(r["mode"], r["K"])][int(r["seed"])] = {k: float(r[k]) for k in KEYS}

print(f"=== per-arm mean (over seeds) | headline={HEAD} ===")
print(f"  {'arm':22} {'K':>4}  " + " ".join(f"{k:>12}" for k in KEYS) + "  (n)")
for k in sorted(by):
    v = list(by[k].values())
    cells = " ".join(f"{statistics.mean(m[key] for m in v):12.4f}" for key in KEYS)
    print(f"  {k[0]:22} {k[1]:>4}  {cells}  ({len(v)})")

def paired(a, b, key):
    A, B = by.get(a, {}), by.get(b, {}); s = sorted(set(A) & set(B))
    if not s: return None
    d = [A[x][key] - B[x][key] for x in s]
    m = statistics.mean(d); sd = statistics.pstdev(d) or 1e-9
    return m, m / (sd / sqrt(len(d))), sum(1 for x in d if x > 0), len(s)

print("\n=== paired vs rotation (Δ>0 = beats rotation), per K ===")
Ks = sorted({k[1] for k in by})
arms = [m for m in {k[0] for k in by} if m != "rotation"]
for kk in Ks:
    print(f"  --- K={kk} ---")
    for arm in sorted(arms):
        for key in KEYS:
            res = paired((arm, kk), ("rotation", kk), key)
            if res:
                print(f"    {arm:22} {key:18} Δ={res[0]:+.4f} t={res[1]:+.2f} ({res[2]}/{res[3]} won)")
