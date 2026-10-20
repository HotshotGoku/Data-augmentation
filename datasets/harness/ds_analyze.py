"""Summarize a ds_downstream TSV: per-arm mean metric + paired (by seed) deltas of each augmenter/real
arm vs the rotation arm. Usage: python ds_analyze.py <tsv>"""
import sys, csv, collections, statistics
from math import sqrt
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
by = collections.defaultdict(dict)  # (tag,mode) -> {seed: metric}
for r in rows: by[(r["tag"], r["mode"])][int(r["seed"])] = float(r["metric"])
print("=== per-arm mean metric ===")
for k in sorted(by):
    v = list(by[k].values()); print(f"  {k[0]:22} {k[1]:11} mean={statistics.mean(v):.4f}  (n={len(v)})")
def paired(a, b):
    A, B = by.get(a, {}), by.get(b, {}); s = sorted(set(A) & set(B))
    if not s: return None
    d = [A[x] - B[x] for x in s]; m = statistics.mean(d); sd = statistics.pstdev(d) or 1e-9
    return m, m / (sd / sqrt(len(d))), sum(1 for x in d if x > 0), len(s)
rot = [k for k in by if k[1] == "rotation"]
print("=== paired vs rotation (Δ>0 = beats rotation) ===")
for a in [k for k in by if k[1] in ("augmenter", "real")]:
    for r in rot:
        res = paired(a, r)
        if res: print(f"  {a[0]}/{a[1]:10} - rotation: Δ={res[0]:+.4f}  t={res[1]:+.2f}  ({res[2]}/{res[3]} seeds won)")
