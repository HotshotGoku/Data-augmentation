"""Build a replicate-pair manifest from a flat image dir: pairs = ordered (src,tgt) within the same label,
src!=tgt, optionally excluding a held-out rep (so the augmenter never sees the downstream test rep).
Env: IMAGES (glob), LABEL_RE (groups 'lab','rep'), EXCLUDE_REP [''], OUT, MAX_PAIRS_PER_LABEL [0=all]."""
import os, glob, re, json, itertools, random
from collections import defaultdict
E = os.environ; LRE = re.compile(E["LABEL_RE"]); EXC = E.get("EXCLUDE_REP", "")
MAXP = int(E.get("MAX_PAIRS_PER_LABEL", "0")); OUT = E["OUT"]
by = defaultdict(list)
for fp in sorted(glob.glob(E["IMAGES"])):
    m = LRE.search(fp)
    if not m: continue
    g = m.groupdict()
    if EXC and (g.get("rep", "") or "") == EXC: continue
    by[g["lab"]].append(fp)
random.seed(0); n = 0
with open(OUT, "w") as o:
    for lab, fs in by.items():
        pairs = list(itertools.permutations(fs, 2))
        if MAXP > 0 and len(pairs) > MAXP:
            random.shuffle(pairs); pairs = pairs[:MAXP]
        for a, b in pairs:
            o.write(json.dumps({"source_path": a, "target_path": b, "prompt": ""}) + "\n"); n += 1
print(f"wrote {n} pairs over {len(by)} labels -> {OUT}")
