# Data-Augmentation project — data / models / metrics / pipeline map

Replicate-to-replicate augmenter (ControlNet + Stable Diffusion 1.5). Repo root on DCC:
`/hpc/group/youlab/sa603/code/Data_augmentation/`. This file maps every dataset, checkpoint,
output, and how to onboard a new dataset. (Living doc; update when adding datasets/results.)

## Datasets
| tag | what | canonical 256px location | train pairs (JSON) | eval reals / sources | notes |
|---|---|---|---|---|---|
| multiplexed | Kinshuk's aTc/IPTG multiplexed sensing | `data/multiplexed_eval/{reals, reals_heldout, sources}` | `data/multiplexed_eval/train_multiplexed.json` | reals_heldout (val+test) ; `sources/{cond}_src.TIF` | reps: train {1,4,5,8,9,10}, val {2,6,11}, test {3,7,12}; cond = `R{row}C{col}`. **Ground-truth labels** (aTc=Row, IPTG=Column): `ks723/storage/Exp_images/Multiplexed_patterning/labels_aTc_IPTG*.json` |
| 2sp (KL) | Kristen's 2-species KL co-culture | `data/ood_kl_2species/`, `data/finetune_2sp/test_2sp` | (2sp finetune data) | test_2sp (held out) | |
| rattray | GaTech Rattray-2023 P. aeruginosa | `data/external_datasets/rattray_2023/` | `.../train_rattray.json` (720 pairs) | `.../test_reals` (48), `.../test_sources` (13 strains) | already bg-removed 256px; 13 strains held out. **Unseen by base AND generalist** -> clean few-shot testbed |
| generalist domains | internal lab colony datasets | `data/generalist/canon/<code>` | `data/generalist/train_generalist.json` (5,722 pairs) | `data/generalist/test/<code>/{sources,reals}` | codes: pakp (EmrahPaKp Pa+Kp), kl2 (KL crops), ff (Final_folder), nlev (NL_evolution), selx (Selected_Exps) + mplex |
| source trove | 399 GB of raw lab experiments | `ks723/storage/Exp_images/` | — | — | many replicate datasets catalogued (EmrahPaKp, NL_evolution, Final_folder, Selected_Exps, KL, multiplexed) |

## Checkpoints (all under `Data_augmentation/`)
- **base:** `lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt`
- **2sp-ft:** `finetune_runs/shallowsweep_lr5e-6/lightning_logs/version_53408286/checkpoints/epoch=3-step=75999.ckpt`
- **rattray-ft:** `rattray_runs/rat_epochcurve/lightning_logs/version_54107493/checkpoints/epoch=3-step=719.ckpt`
- **multiplexed-ft:** `rattray_runs/multiplexed_ft/lightning_logs/version_54549115/checkpoints/epoch=2-step=1448.ckpt`
- **generalist:** `rattray_runs/generalist/lightning_logs/version_54592625/checkpoints/epoch=3-step=5723.ckpt`

## Metrics (metrics_version 2) — `eval_metrics.py` + `cmmd.py` + `persample.py`
- **CMMD** (distribution): CLIP ViT-L/14 @ **336px** (`openai/clip-vit-large-patch14-336`), σ²=10, ×1000, unbiased MMD². Lower = better. Matches the original CMMD paper / Kinshuk.
- **Per-sample realism** vs the *nearest* real replicate (generation is unpaired): `realism_nn_lpips_alex/vgg`, `realism_nn_ssim`, `realism_nn_orb` (Kinshuk's exact functions from `cldm/metrics.py`, copied into `persample.py`), read against `baseline_nn_real_vs_real`.
- **realism_vs_sibling / baseline_real_vs_real** (LPIPS ratio; 1.0 = indistinguishable from real).
- **diversity** (mode-collapse guard), **copy_rate** (memorization; 0 = novel, not copied), **FID** (kept for continuity; unreliable at our n).
- Per model, `eval_metrics.py` writes `metrics.json` + `per_prefix.csv` + `per_image.csv` (per-generated-image row: nearest real + all per-sample scores — the biologist-facing artifact).

## Result outputs
- `metrics_v2_summary.tsv` — all families (base / specialists / generalist / warm-start), CMMD-336 + per-sample.
- `multiplexed_ft_eval_out/`, `generalist_eval_out/`, `mplex_warmstart_eval_out/` — per-model `gen/` + metrics.
- `downstream_multiplexed_out/results.tsv` — aTc/IPTG decoder (classification acc + regression MAE/R²), real vs classical vs augmenter across K.
- `adapt_curve_out/curve_<dataset>.tsv` — data-efficiency curves (quality vs #pairs, base vs generalist).

## Onboard a new dataset (turnkey pipeline)
1. **Canonicalize** images to 256px (pattern: `build_generalist_pairs.py` / `build_multiplexed_pairs.py`).
2. **Build replicate pairs** JSON (`{source_path, target_path}` within each condition): `build_*_pairs.py`.
3. **Fine-tune:** `reptorep_finetune_rattray.py` (env `FT_JSON / FT_RESUME / FT_TAG / FT_LR / FT_EPOCHS / FT_SAVE_TOPK`), via `Slurm_scripts/finetune_rattray_sa603.sh`.
4. **Generate + evaluate:** `reptorep_infer_rattray.py` then `eval_metrics.py --gen_dir <gen> --real_spec "TAG:<reals>"`, via `rescore_metrics_sa603.sh` (eval-only, reuses `gen/`).
- **Kinshuk's format to match for uniformity:** `{id}_{rep}.TIF` (or `Rep{X}_Row{Y}_Column{Z}.TIF`); split folders `Combined_train_set / Combined_validation_set / Combined_test_set`; optional `_AUG100` augmented folders.

## Key scripts
- `eval_metrics.py` (metrics v2), `cmmd.py` (CMMD-336), `persample.py` (Kinshuk's SSIM/LPIPS-VGG/ORB), `rescore_metrics_sa603.sh` (re-score all).
- `reptorep_finetune_rattray.py` (env-driven fine-tune), `reptorep_infer_rattray.py` (generate), `build_*_pairs.py` (dataset builders).
- `downstream_multiplexed_decode.py` (aTc/IPTG decoder), `build_ft_subset.py` + `adapt_curve_sa603.sh` (data-efficiency), `gen_multiplexed_synth.py` (synthetic replicates).
