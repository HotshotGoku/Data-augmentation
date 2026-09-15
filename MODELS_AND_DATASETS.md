# Models and Datasets: definitions and exact composition

Companion to `DATASETS.md` (which maps where files live). This doc defines what each model IS,
what data it saw, what a "pair" means, and how we measure quality. Written for the replicate-to-
replicate augmenter (ControlNet + Stable Diffusion 1.5). All numbers verified against the code/data
on the DCC, not memory (2026-09-15).

## 1. The core idea in one line
Given one real colony image (a replicate), generate new, realistic, DIFFERENT replicates of the same
condition, so a small real dataset can be expanded with synthetic siblings.

## 2. What a "pair" is, and what "N pairs" means
- A training example is a **pair** = (source real replicate -> target real replicate) of the SAME
  condition/strain. The model learns "given sibling A, produce a plausible sibling B." Because A and B
  are different real replicates, it learns the distribution of siblings, not a copy.
- **"N pairs" counts RAW replicate pairs** (N lines in the fine-tune JSON built by `build_ft_subset.py`).
  It is NOT pre-augmented. Example: "adapts at ~100 pairs" means 100 distinct source->target real pairs.
- **Augmentation is applied on the fly**, not pre-generated. `reptorep_dataset_rattray.py` applies a
  random 90-degree rotation + random flip (dihedral) to each pair every time it is loaded
  (`RATTRAY_AUG=1`). Over 8 epochs each pair is seen ~8 times in different orientations. This differs
  from the classic pipeline that physically pre-generates ~100 rotated copies per image (dataset 100x
  larger on disk). Same rotation-invariance goal, different bookkeeping. The base model's 3 datasets,
  by contrast, WERE pre-augmented 100x (the `_AUG100` folders), which is why their pair counts are huge.

## 3. The model family (base, specialists, generalist)
Every model is the same architecture (SD 1.5 backbone frozen via `sd_locked=True`; only the ControlNet
branch trains). They differ only in the data they were trained on.

### base
The foundation model, trained from the ControlNet init (`control_sd15_ini`) for 5 epochs on 3 internal
datasets (already 100x rotation-augmented), 300,600 pairs total:
| folder | plain name | pairs |
|---|---|---|
| EmrahPaKp_dataset_renamed_AUG100_TrainValOnly | PaKp set (Pa + Kp) | 198,800 |
| Final_folder_uniform_fixedseed_100AUG | branching set (paper) | 51,400 |
| NL_evolution_library_..._AUG100 | evolution / mutant library | 50,400 |

(Person-attributions to confirm with Kinshuk: his paper / Jia's PA mutant library / PaKp.)
Checkpoint: `lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt`.
Behavior: strong in-distribution; on a new domain it imposes its learned dense-radial branching prior
and ignores input structure (fails out-of-distribution) until fine-tuned.

### specialists (base + fine-tune on ONE dataset)
Each is the base, fine-tuned a few epochs at low LR on a single dataset. Best model on that dataset.
| specialist | dataset | checkpoint |
|---|---|---|
| 2sp-ft | Kristen 2-species KL co-culture | finetune_runs/shallowsweep_lr5e-6/.../epoch=3-step=75999.ckpt |
| rattray-ft | GaTech Rattray compact P. aeruginosa | rattray_runs/rat_epochcurve/.../epoch=3-step=719.ckpt |
| multiplexed-ft | Kinshuk multiplexed aTc/IPTG dishes | rattray_runs/multiplexed_ft/.../epoch=2-step=1448.ckpt |

### generalist (base + fine-tune on MANY datasets jointly)
The base, jointly fine-tuned on 6 on-target domains at once, 5,722 RAW pairs. NOT "base + Kristen +
Rattray + multiplexed" -- it excludes Rattray and re-includes the base's own 3 sets as canonical pairs.
| domain code | dataset | train pairs |
|---|---|---|
| pakp | EmrahPaKp (Pa+Kp) | 1,762 |
| kl2 | Kristen 2-species | 860 |
| ff | Final_folder (branching) | 410 |
| nlev | NL_evolution | 432 |
| selx | Selected_Exps | 326 |
| mplex | multiplexed aTc/IPTG | 1,932 |
| **total** | | **5,722** |

Checkpoint: `rattray_runs/generalist/lightning_logs/version_54592625/checkpoints/epoch=3-step=5723.ckpt`.
Role: a strong universal DEFAULT (beats base on CMMD on all 6 domains, better diversity), but a
dedicated specialist still beats it on that specialist's own domain. **Rattray is deliberately held out
of the generalist so it is unseen by BOTH base and generalist -> a clean few-shot testbed.**

## 4. Datasets: train / held-out test counts
Held-out = whole conditions/strains kept OUT of pair-building (zero leakage). CMMD and realism are
evaluated on these held-out reals.
| dataset | train pairs | test sources | test reals | notes |
|---|---|---|---|---|
| pakp (EmrahPaKp) | 1,762 | 18 | 68 | |
| kl2 (Kristen 2sp) | 860 | 5 | 36 | |
| ff (Final_folder) | 410 | 28 | 64 | |
| nlev (NL_evolution) | 432 | 12 | 36 | |
| selx (Selected_Exps) | 326 | 24 | 48 | |
| mplex (multiplexed) | 1,932 | 70 | 417 (val+test reps) | reps: train {1,4,5,8,9,10}, val {2,6,11}, test {3,7,12} |
| rattray | 720 | 13 | 48 | 64 train strains, 13 held-out strains; unseen by base+generalist |

Held-out split rule: ~15% of conditions per domain (`HOLDOUT_EVERY=7` in `build_generalist_pairs.py`);
Rattray by whole strain.

## 5. Metrics (what "good" means)
- **CMMD** (distribution match; LOWER is better). CLIP ViT-L/14 @ 336px, sigma^2=10, x1000, unbiased
  MMD^2 (`cmmd.py`). Compares the whole set of generated images to the whole set of real images. More
  stable than FID at our small n. Matches the CMMD paper / Kinshuk. Good = low CMMD.
- **realism_vs_sibling** (per-sample; ratio, LOWER is better, 1.0 = ideal). For each generated image,
  LPIPS to its NEAREST real replicate of the same condition, divided by the real-to-real LPIPS floor
  (`baseline_real_vs_real`). 1.0x = as close to real as two real replicates are to each other; 2.5x =
  far off. Good = near 1.0.
- **per-sample nn metrics** (`realism_nn_lpips_vgg/alex`, `realism_nn_ssim`): individual-sample closeness
  to the nearest real replicate (biologist-facing). vgg/alex LOWER better; ssim HIGHER better.
- **diversity** (mode-collapse guard; HIGHER is better). LPIPS among the model's own samples for one
  input. Good = high (varied siblings); near 0 = mode collapse (bad).
- **copy_rate** (memorization; 0 is best). Fraction of generated images that are near-copies of a
  training image. Ours is 0 everywhere = novel, not memorized (a key credibility point).
- **FID** (kept for continuity; unreliable at our n). LOWER better, but do not over-index on it.

## 6. Checkpoint inventory (for running elsewhere)
All under `/hpc/group/youlab/sa603/code/Data_augmentation/`, world-readable (group `youlab`).
Each ~7.7-8.1G (includes optimizer state).
| model | path | size |
|---|---|---|
| base | lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt | 8.1G |
| 2sp-ft | finetune_runs/shallowsweep_lr5e-6/lightning_logs/version_53408286/checkpoints/epoch=3-step=75999.ckpt | 7.7G |
| rattray-ft | rattray_runs/rat_epochcurve/lightning_logs/version_54107493/checkpoints/epoch=3-step=719.ckpt | 8.1G |
| multiplexed-ft | rattray_runs/multiplexed_ft/lightning_logs/version_54549115/checkpoints/epoch=2-step=1448.ckpt | 8.1G |
| generalist | rattray_runs/generalist/lightning_logs/version_54592625/checkpoints/epoch=3-step=5723.ckpt | 8.1G |

Load: set `FT_EVAL_CKPT=<path>`; `pipeline.py` / `reptorep_infer_*.py` read it (the model is built from
`cldm_v15.yaml` and `load_state_dict` unwraps the Lightning checkpoint). Generate synthetic replicates
with `gen_multiplexed_synth.py` (`SYNTH_CKPT`, `SYNTH_DIR`, `SYNTH_PER_SRC`).

## 7. Onboard a new dataset (turnkey)
1. Canonicalize images to 256px (pattern: `build_generalist_pairs.py` / `build_multiplexed_pairs.py`).
2. Build replicate pairs JSON within each condition, holding out whole conditions/strains for test
   (`build_*_pairs.py`).
3. Fine-tune from base (or generalist): `reptorep_finetune_rattray.py` (env `FT_JSON/FT_RESUME/FT_LR/
   FT_EPOCHS/FT_TAG/FT_SAVE_TOPK`), via `Slurm_scripts/finetune_rattray_sa603.sh`.
4. Generate + evaluate: `reptorep_infer_rattray.py` then `eval_metrics.py --gen_dir <gen> --real_spec
   "TAG:<reals>"`, via `rescore_metrics_sa603.sh` (eval-only, reuses `gen/`).
Kinshuk's naming convention to match: `{id}_{rep}.TIF` or `Rep{X}_Row{Y}_Column{Z}.TIF`; split folders
`Combined_train_set / Combined_validation_set / Combined_test_set`; optional `_AUG100`.
