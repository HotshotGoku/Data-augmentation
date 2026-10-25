"""Stratified oversample of the branching augmenter manifest by a 2x2 size x shape grid.

Branching's label (shape_factor) is CONTINUOUS, so balance_manifest.py's per-condition
oversampling is a no-op here (each ffc condition already has equal reps). The imbalance lives in
the FEATURE distribution: big+branched colonies dominate (263/410 = 64% of pairs) while round and
small colonies are 2-10x under-represented -> the generator regresses small/round sources toward
the branched majority (confirmed: corr(area_frac,shape_factor)=-0.61, corr(area_frac,drift)=+0.54).

Fix (P4 part 3): bin each pair by its TARGET image's (area_frac, shape_factor) into a 2x2
size x shape grid, then oversample each cell up to the densest cell (cyclic, deterministic) --
same oversample-to-max semantics as balance_manifest.py, but keyed on features not condition.

Edges (user-approved 2026-10-25): small = area_frac < 0.06; round = shape_factor > -1.00.
# ponytail: the big+round cell has only ~25 unique imgs -> ~10x duplication = memorization ceiling;
# accepted by the user for maximal drift correction; it will surface in the probe + synth montage.

Usage: python stratify_branching_manifest.py            # DCC defaults
       python stratify_branching_manifest.py --selfcheck
"""
import os, json, csv, argparse
from collections import defaultdict

REPO = "/hpc/group/youlab/sa603/code/Data_augmentation"
DATA = "/hpc/group/youlab/sa603/data"
MANIFEST = f"{DATA}/clean/branching_256/train_branching.json"
FEATURES = f"{REPO}/downstream/outputs/branching_downstream_out/features_branching_clean.csv"
AF_SMALL = 0.06    # area_frac below this = small colony
SF_ROUND = -1.00   # shape_factor above this = round colony


def bin2x2(sf, af):
    s = "small" if af < AF_SMALL else "big"
    r = "round" if sf > SF_ROUND else "branched"
    return f"{s}+{r}"


def oversample(rows, key_of):
    """Duplicate each cell's rows cyclically up to the max cell's count (deterministic)."""
    g = defaultdict(list)
    for r in rows:
        g[key_of(r)].append(r)
    mx = max(len(v) for v in g.values())
    out = []
    for v in g.values():
        out += [v[i % len(v)] for i in range(mx)]
    return out, {k: len(v) for k, v in g.items()}, mx


def load_feats(path):
    return {row["name"]: (float(row["shape_factor"]), float(row["area_frac"]))
            for row in csv.DictReader(open(path))}


def main(manifest, features, out):
    feat = load_feats(features)
    rows = [json.loads(l) for l in open(manifest) if l.strip()]
    miss = [r for r in rows if os.path.basename(r["target_path"]) not in feat]
    assert not miss, f"{len(miss)} target imgs missing from features CSV, e.g. {os.path.basename(miss[0]['target_path'])}"

    def key(r):
        sf, af = feat[os.path.basename(r["target_path"])]
        return bin2x2(sf, af)

    bal, counts, mx = oversample(rows, key)
    with open(out, "w") as o:
        for r in bal:
            o.write(json.dumps(r) + "\n")
    cells = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"{len(rows)} -> {len(bal)} pairs | 2x2 cells (orig): {cells} | -> {mx}/cell -> {out}", flush=True)


def _selfcheck():
    rows = [{"target_path": f"/d/{n}.TIF"} for n in ["a", "b", "c", "d"]]
    feat = {"a": (-1.5, 0.2), "b": (-1.4, 0.2), "c": (-0.5, 0.01), "d": (-1.6, 0.2)}  # 3 big+branched, 1 small+round
    bal, counts, mx = oversample(rows, lambda r: bin2x2(*feat[os.path.basename(r["target_path"]).split(".")[0]]))
    assert mx == 3 and len(bal) == 6, f"expected 2 cells x max 3 = 6, got {len(bal)} (counts {counts})"
    assert bin2x2(-0.5, 0.01) == "small+round" and bin2x2(-1.5, 0.2) == "big+branched"
    print(f"selfcheck OK (3 big+branched + 1 small+round -> both 3; total {len(bal)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); raise SystemExit
    out = a.out or a.manifest.replace(".json", "_stratified.json")
    main(a.manifest, a.features, out)
