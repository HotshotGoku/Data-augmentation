"""Fine-tune the augmenter on the Rattray P. aeruginosa replicate pairs, resuming from our base model.
Same instrumentation as reptorep_finetune.py (seed, FT_FREEZE layer freezing via optimizer-exclusion,
tracking manifest, env-driven LR/epochs/tag/save_top_k) but uses the no-crop Rattray pair loader.

Env: FT_JSON [train_rattray.json], FT_RESUME [CKPT_PATH_V4], FT_LR [5e-6], FT_EPOCHS [6],
     FT_TAG [rat_lr<LR>], FT_SEED [42], FT_FREEZE [none|hint|shallow], FT_SAVE_TOPK [1].
Upload to Data_augmentation repo root.
"""
from Data_augmentation.utils.shared_resources_config import CLDM_V15_YAML  # noqa: sets up cldm path
import os
import types as _types
import torch as _torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from reptorep_dataset_rattray import MyDataset
from cldm.model import create_model, load_state_dict
from Data_augmentation.utils.local_config import CKPT_PATH_V4
from Data_augmentation.utils import tracking

FT_JSON   = os.environ.get("FT_JSON", "/hpc/group/youlab/sa603/data/external_datasets/rattray_2023/train_rattray.json")
FT_RESUME = os.environ.get("FT_RESUME", CKPT_PATH_V4)
FT_LR     = float(os.environ.get("FT_LR", "5e-6"))
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", "6"))
FT_TAG    = os.environ.get("FT_TAG", f"rat_lr{FT_LR:.0e}")
FT_SEED   = int(os.environ.get("FT_SEED", "42"))
FT_FREEZE = os.environ.get("FT_FREEZE", "none").lower()
FT_SAVE_TOPK = int(os.environ.get("FT_SAVE_TOPK", "1"))   # disk-safe default: keep only the last/best
FT_SDLOCKED = os.environ.get("FT_SDLOCKED", "1") == "1"    # 1 = freeze all SD (orig); 0 = also train SD decoder (output_blocks+out)
FT_BATCH  = int(os.environ.get("FT_BATCH", "4"))           # lower to fit memory when sd_locked=0
FT_ACCUM  = int(os.environ.get("FT_ACCUM", "1"))           # grad accumulation (FT_BATCH=2 FT_ACCUM=2 -> effective batch 4)
if FT_FREEZE not in ("none", "hint", "shallow"):
    raise SystemExit(f"[rattray] unknown FT_FREEZE={FT_FREEZE!r}")

print(f"[rattray] resume={FT_RESUME}\n[rattray] json={FT_JSON}\n[rattray] lr={FT_LR} epochs={FT_EPOCHS} tag={FT_TAG} seed={FT_SEED} freeze={FT_FREEZE} save_top_k={FT_SAVE_TOPK} sd_locked={FT_SDLOCKED} batch={FT_BATCH} accum={FT_ACCUM}", flush=True)

pl.seed_everything(FT_SEED, workers=True)

model = create_model(CLDM_V15_YAML).cpu()
model.load_state_dict(load_state_dict(FT_RESUME, location="cpu"))
model.learning_rate = FT_LR
model.sd_locked = FT_SDLOCKED
model.only_mid_control = False

# Layer freezing via optimizer-exclusion (checkpoint-safe; see reptorep_finetune.py for rationale).
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

    def _cfg_opt(self):
        params = [p for p in self.control_model.parameters() if id(p) not in frozen_ids]
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        return _torch.optim.AdamW(params, lr=self.learning_rate)
    model.configure_optimizers = _types.MethodType(_cfg_opt, model)
    print(f"[rattray] FT_FREEZE={FT_FREEZE}: excluded {frozen_params:,} params from optimizer", flush=True)

run_id, run_dir = tracking.write_manifest("finetune_rattray", {
    "tag": FT_TAG, "lr": FT_LR, "epochs": FT_EPOCHS, "batch_size": FT_BATCH, "seed": FT_SEED, "sd_locked": FT_SDLOCKED, "accum": FT_ACCUM,
    "freeze": FT_FREEZE, "frozen_params": frozen_params, "resume_ckpt": FT_RESUME,
    "train_json": FT_JSON, "save_top_k": FT_SAVE_TOPK, "dataset": "rattray_Paeruginosa",
})

dataset = MyDataset(FT_JSON)
dataloader = DataLoader(dataset, num_workers=0, batch_size=FT_BATCH, shuffle=True)
print(f"[rattray] {len(dataset)} pairs", flush=True)

ckpt_cb = ModelCheckpoint(save_top_k=FT_SAVE_TOPK, every_n_epochs=1, filename="{epoch}-{step}")
trainer = pl.Trainer(
    default_root_dir=f"/hpc/group/youlab/sa603/code/Data_augmentation/rattray_runs/{FT_TAG}",
    enable_progress_bar=False, gpus=1, precision=32, max_epochs=FT_EPOCHS, callbacks=[ckpt_cb],
    accumulate_grad_batches=FT_ACCUM,
)
trainer.fit(model, dataloader)
print("[rattray] done", flush=True)
