"""Build a proper HF dataset (two Image columns: target + conditioning) for ControlNet training.
save_to_disk -> loaded via a load_from_disk shim in the training script. Raw imagefolder can't do two
image columns (every image needs a metadata row), hence this."""
import json
from datasets import Dataset, Image, DatasetDict
SRC = "/hpc/group/youlab/sa603/data/branching_256/train_branching.json"
OUT = "/hpc/group/youlab/sa603/data/branching_256/sd3_cn_ds"
rows = [json.loads(l) for l in open(SRC) if l.strip()]
ds = Dataset.from_dict({
    "image": [r["target_path"] for r in rows],            # target replicate
    "conditioning_image": [r["source_path"] for r in rows],  # source replicate
    "text": [""] * len(rows),
}).cast_column("image", Image()).cast_column("conditioning_image", Image())
DatasetDict({"train": ds}).save_to_disk(OUT)
print(f"saved {len(rows)} pairs (2 Image cols) -> {OUT}")
