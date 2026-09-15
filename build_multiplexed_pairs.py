"""Build multiplexed fine-tune data (replicate-to-replicate), leakage-controlled.

Conditions (Row_Column) are shared across splits; replicates (Rep{X}) are split:
  TRAIN reps = {1,4,5,8,9,10}   VAL = {2,6,11}   TEST = {3,7,12}
We fine-tune on TRAIN-rep pairs and evaluate against VAL+TEST reps (unseen in training).

Reads the already-256px canonical images in multiplexed_eval/reals/<cond>_Rep<X>.TIF.
Writes:
  train_multiplexed.json  -- all ordered TRAIN-rep pairs within each condition
  reals_heldout/          -- VAL+TEST reps (the honest sibling pool for eval)
Run on the DCC.
"""
import os, glob, re, json, shutil

MPX = "/hpc/group/youlab/sa603/data/multiplexed_eval"
REALS = MPX + "/reals"
HELDOUT = MPX + "/reals_heldout"; os.makedirs(HELDOUT, exist_ok=True)
TRAIN_REPS = {1, 4, 5, 8, 9, 10}
HELDOUT_REPS = {2, 3, 6, 7, 11, 12}

byc = {}
for fp in glob.glob(REALS + "/*.TIF"):
    m = re.match(r"(R\d+C\d+)_Rep(\d+)\.TIF$", os.path.basename(fp))
    if m:
        byc.setdefault(m.group(1), {})[int(m.group(2))] = fp

pairs = []
for cond, reps in byc.items():
    tr = [reps[r] for r in reps if r in TRAIN_REPS]
    for i in tr:
        for j in tr:
            if i != j:
                pairs.append({"source_path": i, "target_path": j, "prompt": ""})
    for r in reps:
        if r in HELDOUT_REPS:
            shutil.copy(reps[r], HELDOUT + "/" + os.path.basename(reps[r]))

with open(MPX + "/train_multiplexed.json", "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
print(f"conditions: {len(byc)} | train pairs: {len(pairs)} | held-out reals (val+test): {len(glob.glob(HELDOUT + '/*.TIF'))}")
