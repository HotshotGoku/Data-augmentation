"""
Modified version of simtoexp_dataset.py for Data Augmentation project
Uses modules from the original project and custom config
"""

# Import config first to setup paths
from utils import shared_resources_config

import json
import cv2
import numpy as np
import os
import random

from torch.utils.data import Dataset
from cldm.preprocess import preprocess_experimental_backgroundblack
from gen_augment import augment_pair

# GEN_AUG path (Kinshuk feedback 2026-11-16): preprocessing is baked to disk by the builder
# (initialstage_mod / dish_remove), so the dataloader just LOADS + applies runtime rotate/flip/brightness.
# Default off -> legacy behaviour (runtime backgroundblack, no geo-aug) is unchanged.
GEN_AUG = os.environ.get("GEN_AUG", "0") == "1"
AUG_BRIGHTNESS = os.environ.get("AUG_BRIGHTNESS", "1") == "1"   # 0 for fluorescent (mux/kl)
AUG_FILL = (int(os.environ.get("AUG_FILL", "255")),) * 3       # rotation border: 255=white, 0=black


def _load_rgb(path, size=256):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size))
    return img


base_folder = '/hpc/group/youlab/ks723/storage/'  # location of prompt_simtoexp.json


class MyDataset(Dataset):
    def __init__(self, json_file_path=os.path.join(base_folder, 'prompt_experiments_permutations.json')):
        self.data = []
        self.json_file_path = json_file_path

        # Load JSON file - each entry now includes its folder path
        with open(self.json_file_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        folder = item['folder']  # Folder path from JSON entry

        source_path = os.path.join(folder, source_filename)
        target_path = os.path.join(folder, target_filename)

        if GEN_AUG:
            # preprocessing already baked to disk (initialstage_mod / dish_remove) -> just load +
            # apply runtime rotate/flip/brightness with the SAME params to source and target.
            source = _load_rgb(source_path)
            target = _load_rgb(target_path)
            source, target, _ = augment_pair(source, target, np.random.default_rng(),
                                             brightness=AUG_BRIGHTNESS, fill=AUG_FILL)
        else:
            source = preprocess_experimental_backgroundblack(source_path)
            target = preprocess_experimental_backgroundblack(target_path)

        # Already 3 channels (RGB) - no conversion needed for experimental images

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Conditioning dropout (CFG-for-control): blank the hint with prob HINT_DROPOUT so the model
        # learns to generate uncontrolled too -> stronger diversity control at inference. Default 0 = no change.
        _p = float(os.environ.get("HINT_DROPOUT", "0.0"))
        if _p > 0 and random.random() < _p:
            source = np.zeros_like(source)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
