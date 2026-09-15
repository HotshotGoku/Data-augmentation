"""Write a deterministic N-pair subset of a train-pairs JSON, for data-efficiency curves.
Env: FT_JSON_FULL, FT_JSON_SUBSET, SUBSET_N (<=0 or >=len -> all), SUBSET_SEED [0]."""
import os, random
INP = os.environ["FT_JSON_FULL"]
OUT = os.environ["FT_JSON_SUBSET"]
N = int(os.environ["SUBSET_N"])
SEED = int(os.environ.get("SUBSET_SEED", "0"))
pairs = [l for l in open(INP) if l.strip()]
random.Random(SEED).shuffle(pairs)
sub = pairs if (N <= 0 or N >= len(pairs)) else pairs[:N]
with open(OUT, "w") as f:
    f.writelines(sub)
print(f"[subset] {len(sub)}/{len(pairs)} pairs -> {OUT}")
