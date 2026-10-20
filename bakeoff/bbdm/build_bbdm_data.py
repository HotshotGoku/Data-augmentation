"""Convert our multiplexed paired JSONL into BBDM's A/B directory layout.

BBDM's CustomAlignedDataset pairs A (source/condition) and B (target) by SORTED-INDEX position,
NOT by filename match -- so each pair must occupy the same sorted position in both dirs. We
guarantee that by giving each pair an identical zero-padded filename in A and B.

Layout produced under OUT:
  train/A/<k>.png , train/B/<k>.png   (one per JSONL line; a source repeated across pairs teaches one-to-many)
  val/A , val/B                        (small held-out slice so BBDM's val loop has data)
  test/A/<srcid>.png , test/B/<srcid>.png  (the 70 frozen eval sources; A==B filename so BBDM's
                                            per-source output folder id == source id; B is a placeholder)
Run in the bbdm env (needs opencv). Lightweight I/O -- run on a CPU node, not login-compute."""
import os, json, glob, cv2, random

TRAIN_JSON = "/hpc/group/youlab/sa603/data/multiplexed_eval/train_multiplexed.json"
SOURCES = "/hpc/group/youlab/sa603/data/multiplexed_eval/sources"
OUT = "/hpc/group/youlab/sa603/data/bbdm_multiplexed"
VAL_N = 64
random.seed(42)

def save256(src_path, dst_path):
    im = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"failed to read {src_path}")
    if im.shape[:2] != (256, 256):
        im = cv2.resize(im, (256, 256))
    cv2.imwrite(dst_path, im)  # color-preserving round trip; BBDM reads via PIL

for sub in ["train/A", "train/B", "val/A", "val/B", "test/A", "test/B"]:
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)

pairs = [json.loads(l) for l in open(TRAIN_JSON)]
val_idx = set(random.sample(range(len(pairs)), min(VAL_N, len(pairs))))
tr = va = 0
for k, p in enumerate(pairs):
    split = "val" if k in val_idx else "train"
    fn = f"{k:06d}.png"
    save256(p["source_path"], f"{OUT}/{split}/A/{fn}")
    save256(p["target_path"], f"{OUT}/{split}/B/{fn}")
    tr += split == "train"; va += split == "val"
print(f"[bbdm data] train={tr} val={va} (from {len(pairs)} pairs)", flush=True)

srcs = sorted(glob.glob(f"{SOURCES}/*.TIF"))
for fp in srcs:
    sid = os.path.basename(fp).split("_")[0]      # e.g. R10C1
    save256(fp, f"{OUT}/test/A/{sid}.png")
    save256(fp, f"{OUT}/test/B/{sid}.png")         # placeholder, ignored for generation
print(f"[bbdm data] test sources={len(srcs)} -> {OUT}/test/{{A,B}}", flush=True)
print(f"[bbdm data] DONE -> {OUT}", flush=True)
