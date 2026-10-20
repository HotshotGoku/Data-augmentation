# Data_augmentation

Replicate-to-replicate colony-image augmenter (ControlNet + Stable Diffusion 1.5): given one real
colony replicate, generate new realistic replicates of the same condition to expand scarce datasets.
Spinoff of `Physics_constrained_DL_pattern_prediction` (Kinshuk Sahu et al.).

**Core finding:** no newer backbone beats in-domain SD1.5; the real win is conditioning-dropout SD1.5
(smarter diversity on the same backbone). Full story: `docs/PROJECT_UPDATE.md` + `docs/METHODOLOGY.md`.

## Layout
| Folder | What |
|---|---|
| `pipeline/`    | Shared SD1.5 augmenter engine — train / finetune / infer / metrics / utils |
| `datasets/`    | Per-dataset builders, pilots, onboarding; generic harness; data-efficiency curves |
| `bakeoff/`     | Backbone search (Flux/PixCell/DDPM/GAN/SD3-SDXL/BBDM/SD1.5) — **CLOSED, negative** |
| `downstream/`  | Downstream-utility experiments (aTc/IPTG decoder, colony-feature regression) |
| `diversity/`   | Diversity improvement (CADS, conditioning-dropout, sweeps) |
| `reporting/`   | Figure/deck builders + committed figures + master result tables |
| `experiments/` | Experiment-tracking ledger (`runs.tsv` + per-run manifests) |
| `docs/`        | Methodology, models/datasets reference, literature survey |

## Setup
`conda activate pytorch_PA_patternprediction`. Shared checkpoints/datasets resolve via
`pipeline/utils/shared_resources_config.py` (set `PHYSICS_DL_PROJECT_PATH`). Checkpoints, training
runs, and generated images are gitignored.
