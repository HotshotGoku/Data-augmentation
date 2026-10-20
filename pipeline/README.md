# pipeline/ — shared augmenter engine
Imported/called by every initiative. Do not duplicate this logic elsewhere.
- `reptorep_*.py` — replicate-to-replicate ControlNet train/finetune/dataset/infer. The `*_rattray`
  variants are the **canonical env-driven finetune/infer** used to onboard ANY new dataset.
- `pipeline.py` + `*_reptorep_sa603.sh` / `ControlNet_exp_*` — core launchers.
- `metrics/` — CMMD (`cmmd.py`), per-sample realism (`persample.py`), `eval_metrics.py` + re-score shells.
- `augmentation_baseline/` — classical rotation/flip baseline. `postprocess/` — dish standardization.
- `diagnostics/` — VAE recon sanity. `setup/` — model download / env / wandb. `utils/` — shared config, tracking.
