"""
Fine-tune the replicate-to-replicate augmenter on ONE new dataset, resuming from an existing
checkpoint. Adapted from reptorep_train.py.

Built for hyperparameter sweeps:
  - learning rate, epochs, training JSON, and the checkpoint to resume FROM all come from env vars.
  - a checkpoint is saved AFTER EVERY EPOCH (save_top_k=-1), so a single run gives you the whole
    "epochs" axis to evaluate later — no separate run per epoch count needed.

Env vars (defaults in brackets):
  FT_JSON    [.../finetune_2sp/train_2sp.json]  training pairs (test samples already excluded)
  FT_RESUME  [CKPT_PATH_V4]                      checkpoint to fine-tune FROM (our full-data model)
  FT_LR      [5e-6]                              learning rate
  FT_EPOCHS  [4]                                 max epochs
  FT_TAG     [lr<FT_LR>]                         run label -> finetune_runs/<tag>/
"""
from Data_augmentation.utils.shared_resources_config import CLDM_V15_YAML  # noqa: sets up cldm path
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from reptorep_dataset import MyDataset
from cldm.model import create_model, load_state_dict
from Data_augmentation.utils.local_config import CKPT_PATH_V4
from Data_augmentation.utils import tracking

FT_JSON   = os.environ.get("FT_JSON",   "/hpc/group/youlab/sa603/data/finetune_2sp/train_2sp.json")
FT_RESUME = os.environ.get("FT_RESUME", CKPT_PATH_V4)
FT_LR     = float(os.environ.get("FT_LR", "5e-6"))
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", "4"))
FT_TAG    = os.environ.get("FT_TAG", f"lr{FT_LR:.0e}")
FT_SEED   = int(os.environ.get("FT_SEED", "42"))
FT_FREEZE = os.environ.get("FT_FREEZE", "none").lower()   # none | hint | shallow
FT_SAVE_TOPK = int(os.environ.get("FT_SAVE_TOPK", "-1"))  # -1 = keep every epoch (score-then-prune externally); e.g. 1 = disk-safe
if FT_FREEZE not in ("none", "hint", "shallow"):
    raise SystemExit(f"[finetune] unknown FT_FREEZE={FT_FREEZE!r} (use none|hint|shallow)")

print(f"[finetune] resume : {FT_RESUME}")
print(f"[finetune] json   : {FT_JSON}")
print(f"[finetune] lr={FT_LR}  epochs={FT_EPOCHS}  tag={FT_TAG}  seed={FT_SEED}  freeze={FT_FREEZE}  save_top_k={FT_SAVE_TOPK}", flush=True)

pl.seed_everything(FT_SEED, workers=True)

# Build the model and load OUR trained weights (load_state_dict unwraps the Lightning ckpt).
model = create_model(CLDM_V15_YAML).cpu()
model.load_state_dict(load_state_dict(FT_RESUME, location="cpu"))
model.learning_rate = FT_LR
model.sd_locked = True
model.only_mid_control = False

# --- Layer freezing (Kinshuk: freeze some ControlNet layers so fine-tuning doesn't jumble
# everything). This model uses gradient checkpointing (use_checkpoint=True); its custom
# CheckpointFunction.backward runs torch.autograd.grad over each block's params and ERRORS if any
# has requires_grad=False ("One of the differentiated Tensors does not require grad"). So we do NOT
# flip requires_grad (that would keep the grad graph identical to the working full run); instead we
# EXCLUDE the frozen params from the optimizer via a configure_optimizers override -> they still get
# grads (checkpoint-safe) but are never stepped, so never change. sd_locked=True already freezes the
# SD UNet. ---
import types as _types
import torch as _torch
frozen_ids, frozen_params = set(), 0
if FT_FREEZE != "none":
    cm = model.control_model
    frozen_mods = [cm.input_hint_block]
    if FT_FREEZE == "shallow":
        frozen_mods += [cm.input_blocks[i] for i in range(6)]
    for m in frozen_mods:
        for p in m.parameters():
            if id(p) not in frozen_ids:
                frozen_ids.add(id(p)); frozen_params += p.numel()

    def _configure_optimizers_frozen(self):
        params = [p for p in self.control_model.parameters() if id(p) not in frozen_ids]
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        return _torch.optim.AdamW(params, lr=self.learning_rate)
    model.configure_optimizers = _types.MethodType(_configure_optimizers_frozen, model)
    print(f"[finetune] FT_FREEZE={FT_FREEZE}: excluding {len(frozen_mods)} module groups "
          f"({frozen_params:,} params) from optimizer (checkpoint-safe)", flush=True)
else:
    print("[finetune] FT_FREEZE=none (full ControlNet trainable)", flush=True)
trainable_params = sum(p.numel() for p in model.control_model.parameters()) - frozen_params
print(f"[finetune] optimizer trains ~{trainable_params:,} control_model params", flush=True)

# Durable file-based run record (W&B is synced separately by sync_wandb.py).
run_id, run_dir = tracking.write_manifest("finetune", {
    "tag": FT_TAG, "lr": FT_LR, "epochs": FT_EPOCHS, "batch_size": 4, "seed": FT_SEED,
    "freeze": FT_FREEZE, "frozen_params": frozen_params, "trainable_params": trainable_params,
    "resume_ckpt": FT_RESUME, "train_json": FT_JSON, "save_top_k": FT_SAVE_TOPK,
    "sd_locked": True, "only_mid_control": False,
})

dataset = MyDataset(json_file_path=FT_JSON)
dataloader = DataLoader(dataset, num_workers=0, batch_size=4, shuffle=True)

# One checkpoint per epoch -> the whole "epochs" axis from a single run.
ckpt_cb = ModelCheckpoint(save_top_k=FT_SAVE_TOPK, every_n_epochs=1, filename="{epoch}-{step}")

trainer = pl.Trainer(
    default_root_dir=f"/hpc/group/youlab/sa603/code/Data_augmentation/finetune_runs/{FT_TAG}",
    enable_progress_bar=False, gpus=1, precision=32,
    max_epochs=FT_EPOCHS, callbacks=[ckpt_cb],
)
trainer.fit(model, dataloader)
print("[finetune] done", flush=True)
