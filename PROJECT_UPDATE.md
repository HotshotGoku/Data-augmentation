# Replicate Augmenter — Project Update (2026-10-12)
Sarthak. Current state and verdict; the backbone search is closed. (A detailed writeup of past methodologies will be a separate doc.)

## Bottom line
We ran an exhaustive search for a generative backbone that beats the in-domain SD1.5 replicate-to-replicate augmenter. **Nothing does.** Not Flux (escalated all the way to a bf16 full base with 3× ControlNet capacity), not FastGAN, not StyleGAN2-ADA, not SD3.5-ControlNet, not zero-shot general priors, and not a from-scratch conditional DDPM. Every alternative loses to in-domain SD1.5 on realism and CMMD.

The real, defensible win is **on the SD1.5 side, not from a newer backbone**: training SD1.5 with **conditioning-dropout** to produce more diverse synthetic replicates **beats plain rotation on the metric that matters**, axis-selectively, across two domains — branching → `shape_factor`, multiplexed → `aTc` decode. The one-line summary: *"smarter diversity on SD1.5" wins; "newer backbone" never does.*

## The backbone comparison (branching, image metrics)
Identical scoring for every candidate (`eval_metrics.py`: realism-vs-sibling and diversity via LPIPS, CMMD via CLIP ViT-L/14@336), scored against the same held-out branching reals. Lower realism/CMMD is better; higher diversity is better. Real-vs-real floor realism ≈ 0.25.

| Backbone | realism ↓ | CMMD ↓ | diversity ↑ | Verdict |
|---|---|---|---|---|
| **SD1.5 in-domain — conditioning-dropout (sweet spot s1.0/g7)** | **0.333** | **3.20** | **0.271** | best overall |
| SD1.5 in-domain — baseline | 0.324 | 3.38 | 0.252 | the reliable workhorse |
| Flux — bf16 + CN6 (max escalation) | 0.503 | 4.84 | 0.210 | loses (realism plateau) |
| Flux — scaled fp8 / CN4 | 0.499 | 5.53 | 0.208 | loses |
| Flux — pilot | 0.524 | 8.78 | 0.270 | loses |
| conditional DDPM — from scratch | 0.538 | 9.81 | 0.213 | loses (blurry) |
| FastGAN — lightweight-gan, unconditional | — | 15.4 | 0.271 (global) | loses |
| PixCell — frozen cell-microscopy DiT | washed | ~49.8 | low | loses on branching |
| zero-shot SDXL — img2img, no training | 0.722 | 71.0 | 0.597* | catastrophic |

\*The high "diversity" of the zero-shot prior is hallucinated natural-image texture, not useful biological variation. The 250-step DDIM re-sample of the conditional DDPM (0.542 / 10.11) was no better than 50-step — the blur is structural, not undersampling.

**Flux escalation ladder (the fairest shot a newer backbone got):** pilot `0.524 / 8.78` → scaled fp8/CN4 `0.499 / 5.53` → bf16 full base + CN6 `0.503 / 4.84`. CMMD improves monotonically with each lever but **realism plateaus at ~0.50 and never approaches SD1.5's 0.32**. The gap is structural (faint, incomplete branches), not quantization or undertraining.

## The backbone comparison (multiplexed, image CMMD)
The verdict flips by domain on raw image quality — PixCell's cell-microscopy prior suits fluorescence colonies.

| Backbone | CMMD ↓ | Note |
|---|---|---|
| PixCell ep30 | **5.78** | best on multiplexed |
| SD1.5 in-domain — conditioning-dropout | 7.97 | **beats in-domain baseline** (new, 2026-10-12) |
| SD1.5 in-domain — baseline | 12.6 | |
| zero-shot SDXL | 42.99 | catastrophic (new, 2026-10-12) |
| Flux / conditional DDPM / FastGAN | *in progress* | multiplexed fill-out jobs running |

PixCell wins multiplexed on image CMMD but is useless on branching and was never the downstream winner, so it is a domain-specific curiosity, not a general replacement.

## The real win: downstream utility
Image beauty is not the goal — downstream utility is. The mechanism is about information: **rotation is label-invariant**, so rotated copies carry the *same* label and add no new signal; **synthetic replicates have genuinely different morphology** with their own computed labels, which is new signal on the axis where the synthetic variation is faithful.

- **Branching → `shape_factor`:** conditioning-dropout SD1.5 beats plain rotation. Diverse config (s0.5/g9): shape_factor R² 0.396 vs 0.339 (+0.057, t=2.49, significant in 12/15 seeds). Realistic sweet-spot config (s1.0/g7): +0.052 (t=2.14, 11/15, p≈0.05). Aggregate R² only ties rotation — the win is the morphology axis specifically.
- **Multiplexed → `aTc` decode:** +0.045 row accuracy (t=6.77, 14/15 seeds). But **loses** on `IPTG` (−0.022, t=−2.19) — axis-selective, the same shape as branching.

**Refined claim:** generative augmentation beats rotation on the axis where its synthetic variation is *faithful*, and can *hurt* where it isn't. Axis-selective, not uniform — more defensible and more publishable than "it helps."

## Dish standardization (presentation only)
Generated petri dishes came out inconsistent in size and ~12% inflated. The fix is a post-process (`dish_composite.py`), not a retrain: detect each dish, rescale its rim to the measured real radius, composite only the colony interior inside a fixed real-glass template. **Dual-track and important:** composited images are for figures/slides only — feeding them into downstream *hurts* it, because the fixed-radius feathering clips exactly the shape diversity that is the augmenter's edge. Science uses raw synth; presentation uses composited.

## Methods and scope caveats
- Two domains: branching (dendritic, petri dish; downstream = morphology features) and multiplexed (cell-like fluorescence, no dish; downstream = aTc/IPTG decode).
- Downstream harness: `downstream_colony_features.py` (branching, ResNet50, auto-labeler `colony_features.py`) and `downstream_multiplexed_decode.py` (multiplexed). Paired seeds, data-scarce regimes, held-out real test.
- The multiplexed conditioning-dropout model was trained fresh from the `control_sd15_ini` seed, so its 7.97 CMMD is not a perfectly clean "in-domain + dropout" delta versus the 12.6 baseline (different base recipe).
- Zero-shot screening of general priors does **not** discriminate on a domain this narrow — every untrained backbone is ~equally useless. Real comparison requires trained candidates (GPU-hours each).
- Literature context: whether generative augmentation beats classical augmentation at all in the low-data regime is an open question — "nothing beats rotation on aggregate" is a known phenomenon, not a bug in our setup.

## Key artifacts (DCC: `/hpc/group/youlab/sa603/code/Data_augmentation/`)
- **Backbone pipelines:** `screen_img2img.py` (zero-shot screen), `gan_train_sa603.sh` + `gan_generate.py` + `cmmd_direct.py` (FastGAN), `stylegan2ada_sa603.sh`, `train_controlnet_sd3*.py` + `sd3_*.sh` (SD3.5), `conditional_ddpm.py` + `cond_ddpm_sa603.sh` (DDPM), `flux_{train_controlnet,gen_eval}.py` + `flux_scaled_*/flux_bf16_*` (Flux escalation).
- **SD1.5 wins:** `branching_sd15_train_sa603.sh` (conditioning-dropout via `reptorep_dataset.py` `HINT_DROPOUT`), `sd15_divgen.py` + `sd15_divsweep_sa603.sh` (diversity frontier), `div_downstream_sa603.sh` (downstream).
- **Multiplexed fill-out:** `mux_*_sa603.sh` (sdxl / sd15-dropout / flux / cond-ddpm / fastgan), all reusing `train_multiplexed.json` and scored `MPLEX:multiplexed_eval/reals_heldout`.
- **Dish + figures:** `dish_composite.py`, `dish_normalize.py`, `make_backbone_matrix.py`, `make_crossbackbone.py`.
- **Result tables:** `branching_downstream_out/results_15seed.tsv`, `results_dropout*.tsv`, `results_div*.tsv`; `downstream_multiplexed_out/results_mux15_k1.tsv`.
- Figures and HTML viewers (`backbone_viewer/story.html`, `matrix.html`) and decks are kept local (regenerable; gitignored).
