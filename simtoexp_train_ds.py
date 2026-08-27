"""Downstream sim->exp training for the utility experiment (background-black space).
Trains from the ControlNet init (control_sd15_ini) on a given JSON (real-only or real+synthetic),
so A and B differ ONLY in their training data. Env: DS_JSON (required), DS_TAG, DS_EPOCHS(5),
DS_SEED(42), DS_LR(1e-5), DS_SAVE_TOPK(1). Run on a GPU node. Upload to Data_augmentation root."""
from Data_augmentation.utils.shared_resources_config import CLDM_V15_YAML, CONTROL_SD15_CKPT
from Data_augmentation.utils import tracking
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from simtoexp_dataset_ds import MyDataset
from cldm.model import create_model, load_state_dict

DS_JSON = os.environ["DS_JSON"]
DS_TAG = os.environ.get("DS_TAG", "ds")
DS_EPOCHS = int(os.environ.get("DS_EPOCHS", "5"))
DS_SEED = int(os.environ.get("DS_SEED", "42"))
DS_LR = float(os.environ.get("DS_LR", "1e-5"))
DS_TOPK = int(os.environ.get("DS_SAVE_TOPK", "1"))
ROOT = f"/hpc/group/youlab/sa603/code/Data_augmentation/downstream_runs/{DS_TAG}"

pl.seed_everything(DS_SEED, workers=True)

model = create_model(CLDM_V15_YAML).cpu()
model.load_state_dict(load_state_dict(CONTROL_SD15_CKPT, location="cpu"))
model.learning_rate = DS_LR
model.sd_locked = True
model.only_mid_control = False

ds = MyDataset(DS_JSON)
dl = DataLoader(ds, num_workers=0, batch_size=4, shuffle=True)
print(f"[simexp_ds] tag={DS_TAG} pairs={len(ds)} epochs={DS_EPOCHS} seed={DS_SEED} lr={DS_LR}", flush=True)
tracking.write_manifest("simexp_downstream", {
    "tag": DS_TAG, "json": DS_JSON, "n_pairs": len(ds), "epochs": DS_EPOCHS,
    "seed": DS_SEED, "lr": DS_LR, "resume": "control_sd15_ini", "save_top_k": DS_TOPK,
})

ckpt_cb = ModelCheckpoint(save_top_k=DS_TOPK, every_n_epochs=1, filename="{epoch}-{step}")
trainer = pl.Trainer(default_root_dir=ROOT, enable_progress_bar=False, gpus=1, precision=32,
                     max_epochs=DS_EPOCHS, callbacks=[ckpt_cb])
trainer.fit(model, dl)
print("[simexp_ds] done", flush=True)
