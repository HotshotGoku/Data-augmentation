"""Rattray replicate-pair dataset. Images are background-removed (dark colony on WHITE), already
baked to 256px. Two aug paths:

  GEN_AUG=1 (Kinshuk 2026-11-17): the builder bakes the 0.865 INSCRIBE (content scaled onto a white
    canvas so colony edges stay inside the rotation-safe radius) -> here we LOAD + apply fine
    rotate/flip/brightness via augment_pair with WHITE fill (AUG_FILL=255). Edge/strain morphology
    is preserved through arbitrary rotation.
  default (legacy): clean dihedral (random 90deg + flip) -- no fine rotation, no fill needed. Kept so
    the broaden two-stage launcher (swarmevo stage) is unchanged. RATTRAY_AUG=0 disables.

src/tgt are two different replicates (NOT pixel-aligned), so each path's transform choice is fine.
Returns dict(jpg=target[-1,1], txt=prompt, hint=source[0,1]) -- matches reptorep_dataset.MyDataset.
"""
import json, os, random
import cv2
import numpy as np
from torch.utils.data import Dataset
from gen_augment import augment_pair

GEN_AUG = os.environ.get("GEN_AUG", "0") == "1"
AUG_BRIGHTNESS = os.environ.get("AUG_BRIGHTNESS", "1") == "1"
AUG_FILL = (int(os.environ.get("AUG_FILL", "255")),) * 3   # rattray = white background
AUG = os.environ.get("RATTRAY_AUG", "1") == "1"            # legacy dihedral gate (non-GEN_AUG path)


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
        if GEN_AUG:
            # inscribe already baked to disk -> fine rotate/flip/brightness, white fill.
            src, tgt, _ = augment_pair(src, tgt, np.random.default_rng(),
                                       brightness=AUG_BRIGHTNESS, fill=AUG_FILL)
        elif AUG:
            src, tgt = _dihedral(src), _dihedral(tgt)
        src = src.astype(np.float32) / 255.0             # hint in [0,1]
        tgt = (tgt.astype(np.float32) / 127.5) - 1.0      # target in [-1,1]
        return dict(jpg=tgt, txt=it.get("prompt", ""), hint=src)
