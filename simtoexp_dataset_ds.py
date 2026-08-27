"""Downstream sim->exp dataset for the utility experiment (background-black space).

Reads a JSON (one record per line) with absolute paths + a `synthetic` flag:
  {"source_path": <sim .TIF>, "target_path": <exp .TIF or synth .png>, "synthetic": 0|1, "prompt": ""}

- source (simulation): preprocess_simulation_graybackground -> 3ch -> [0,1]
- target REAL (synthetic=0): preprocess_experimental_backgroundblack -> [-1,1]
- target SYNTHETIC (synthetic=1): augmenter output is already background-black 256px -> passthrough -> [-1,1]

Upload to Data_augmentation repo root.
"""
import json
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from cldm.preprocess import preprocess_experimental_backgroundblack, preprocess_simulation_graybackground


class MyDataset(Dataset):
    def __init__(self, json_file_path=None):
        json_file_path = json_file_path or os.environ["DS_JSON"]
        self.data = []
        with open(json_file_path, "rt") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source = preprocess_simulation_graybackground(item["source_path"])
        source = np.repeat(source[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0

        if int(item.get("synthetic", 0)):
            t = cv2.imread(item["target_path"], cv2.IMREAD_COLOR)
            t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
            t = cv2.resize(t, (256, 256))
        else:
            t = preprocess_experimental_backgroundblack(item["target_path"])
        target = (t.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=item.get("prompt", ""), hint=source)
