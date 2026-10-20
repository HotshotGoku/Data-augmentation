"""Rattray replicate-pair dataset (no cropping — images are already background-removed and fill the
frame). Reads a JSON of {source_path, target_path, prompt} absolute paths. Applies clean dihedral
augmentation (random 90deg rotations + flip, same group element per image, no black-corner artifacts)
to expand the ~800 replicate pairs. Set RATTRAY_AUG=0 to disable.

Returns dict(jpg=target[-1,1], txt=prompt, hint=source[0,1]) — matches reptorep_dataset.MyDataset.
Upload to Data_augmentation repo root.
"""
import json, os, random
import cv2
import numpy as np
from torch.utils.data import Dataset

AUG = os.environ.get("RATTRAY_AUG", "1") == "1"


def _dihedral(img):
    k = random.randint(0, 3)
    img = np.rot90(img, k).copy()
    if random.random() < 0.5:
        img = np.fliplr(img).copy()
    return img


class MyDataset(Dataset):
    def __init__(self, json_file_path):
        self.data = [json.loads(l) for l in open(json_file_path)]

    def __len__(self):
        return len(self.data)

    def _load(self, path):
        im = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return cv2.resize(im, (256, 256))

    def __getitem__(self, idx):
        it = self.data[idx]
        src, tgt = self._load(it["source_path"]), self._load(it["target_path"])
        if AUG:
            src, tgt = _dihedral(src), _dihedral(tgt)
        src = src.astype(np.float32) / 255.0             # hint in [0,1]
        tgt = (tgt.astype(np.float32) / 127.5) - 1.0      # target in [-1,1]
        return dict(jpg=tgt, txt=it.get("prompt", ""), hint=src)
