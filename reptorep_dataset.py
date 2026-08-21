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

from torch.utils.data import Dataset
from cldm.preprocess import preprocess_experimental_backgroundblack


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

        source_path = os.path.join(folder, source_filename)
        target_path = os.path.join(folder, target_filename)

        source = preprocess_experimental_backgroundblack(source_path)
        target = preprocess_experimental_backgroundblack(target_path)

        # Already 3 channels (RGB) - no conversion needed for experimental images
    
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
