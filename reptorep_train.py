"""
Modified version of simtoexp_train.py for Data Augmentation project
Uses shared resources from the original project via shared_resources_config
"""

# Import config first to setup paths
from shared_resources_config import CONTROL_SD15_CKPT, CLDM_V15_YAML

import os, torch
import pytorch_lightning as pl
from torch.utils.data import Subset, DataLoader
from torchvision.utils import make_grid, save_image
from reptorep_dataset import MyDataset
from cldm.logger_custom import ImageLogger
from cldm.model import create_model, load_state_dict
# from local_config import JSON_FILE_3DATASETS_100000
from local_config import JSON_FILE_3DATASETS_FULL


# Configs
resume_path = CONTROL_SD15_CKPT  # Use shared checkpoint from original project
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
# name_suffix= '3datasets_100000'
name_suffix= '3datasets_full_resfix'


# build a fixed mini-set once (K samples)
K = 5
fixed_ds = Subset(MyDataset(json_file_path=JSON_FILE_3DATASETS_FULL), list(range(K)))
fixed_dl = DataLoader(fixed_ds, batch_size=K, shuffle=False, num_workers=0)
fixed_batch = next(iter(fixed_dl))  # dict like {'jpg','hint','txt'}


# minimal callback
class FixedEval(pl.Callback):
    def __init__(self, batch, every=1000, outdir=f"eval_fixed_reptorep_train_{name_suffix}", steps=30, eta=0.0, scale=9.0):
        self.b, self.every, self.outdir = batch, every, outdir
        self.steps, self.eta, self.scale = steps, eta, scale
        os.makedirs(outdir, exist_ok=True)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gs = trainer.global_step
        if gs == 0 or gs % self.every:  # log exactly at multiples
            return
        was_train = pl_module.training
        pl_module.eval()

        # move fixed batch to device
        b = {k: (v.to(pl_module.device) if hasattr(v, "to") else v) for k, v in self.b.items()}

        # use model's existing visualizer
        imgs = pl_module.log_images(
            b, N=K, ddim_steps=self.steps, eta=self.eta,
            unconditional_guidance_scale=self.scale
        )  # dict[str] -> BCHW in [-1,1]

        for name, t in imgs.items():
            if t.ndim != 4: continue
            x = (t.clamp(-1,1) + 1) / 2.0
            grid = make_grid(x, nrow=K)  # one row of K images
            save_image(grid, os.path.join(self.outdir, f"{gs:07d}_{name}.png"))

        if was_train: pl_module.train()


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(CLDM_V15_YAML).cpu()  # Use shared YAML config
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset(json_file_path=JSON_FILE_3DATASETS_FULL)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
fixed_cb = FixedEval(fixed_batch, every=300, outdir=f"eval_fixed_reptorep_train_{name_suffix}", steps=30, eta=0.0, scale=9.0)

# Trainer
trainer = pl.Trainer(enable_progress_bar=False, gpus=1, precision=32, callbacks=[logger, fixed_cb],max_epochs=5)

# Train!
trainer.fit(model, dataloader)
