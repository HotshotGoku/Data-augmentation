"""Build a JOINT broadening train set = Rattray replicate pairs + Myxococcus time-lapse pairs.
Myxococcus (reliable CC0, EBI BioImage Archive S-BIAD2328 WT set) is the only reliably-downloadable
in-scope 'microbial spatial growth' set (the on-modality bacterial sets SwarmEvo/AGAR/urine are all
sign-up/token gated). It is aggregation microscopy — DISTANT morphology from Rattray colonies — so
this is a broadening-on-distant-data test. Consecutive frames within a scope form growth pairs.

Downloads the WT zip if absent, builds canonical 256px 3ch pairs, subsamples to ~match Rattray,
writes train_joint_rattray_myxo.json = rattray pairs + myxo pairs. Run on a compute node.
"""
import os, glob, json, zipfile, subprocess, random
import cv2

BASE = "/hpc/group/youlab/sa603/data/external_datasets"
MYXO = os.path.join(BASE, "myxococcus"); PROC = os.path.join(MYXO, "processed")
RAT_JSON = os.path.join(BASE, "rattray_2023", "train_rattray.json")
os.makedirs(PROC, exist_ok=True)
random.seed(42)

ZIP = os.path.join(MYXO, "wt_images.zip")
URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/328/S-BIAD2328/Files/dataset/WT/images.zip"
if not os.path.exists(ZIP):
    print("[myxo] downloading WT images.zip ...", flush=True)
    subprocess.run(["curl", "-sL", "--max-time", "600", "-o", ZIP, URL], check=True)
if not glob.glob(os.path.join(MYXO, "images", "**", "*.jpg"), recursive=True):
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(MYXO)
print("[myxo] extracted", flush=True)

# group frames by scope dir, sort, make consecutive pairs (i, i+step)
scopes = {}
for fp in glob.glob(os.path.join(MYXO, "images", "**", "*.jpg"), recursive=True):
    scopes.setdefault(os.path.dirname(fp), []).append(fp)

def canon(fp, out):
    a = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(out, cv2.cvtColor(cv2.resize(a, (256, 256)), cv2.COLOR_GRAY2BGR))

myxo_pairs, k = [], 0
STEP = 20  # frames ~1min apart; step 20 -> ~20min growth delta (visible change, not identical)
for sd, frames in scopes.items():
    frames.sort()
    for i in range(0, len(frames) - STEP, STEP):
        s_out = os.path.join(PROC, f"m{k}_s.png"); t_out = os.path.join(PROC, f"m{k}_t.png")
        canon(frames[i], s_out); canon(frames[i + STEP], t_out)
        myxo_pairs.append({"source_path": s_out, "target_path": t_out, "prompt": ""}); k += 1

# subsample myxo pairs to ~match rattray count (balance the joint set)
rat_pairs = [json.loads(l) for l in open(RAT_JSON)]
random.shuffle(myxo_pairs)
myxo_pairs = myxo_pairs[:len(rat_pairs)]
print(f"[myxo] {len(myxo_pairs)} myxo pairs (balanced to {len(rat_pairs)} rattray)", flush=True)

joint = rat_pairs + myxo_pairs
random.shuffle(joint)
outp = os.path.join(BASE, "rattray_2023", "train_joint_rattray_myxo.json")
with open(outp, "w") as f:
    for r in joint:
        f.write(json.dumps(r) + "\n")
print(f"[joint] {len(joint)} pairs -> {outp}", flush=True)
