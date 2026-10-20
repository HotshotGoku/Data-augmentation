# Methodology — decisions and rationale

A map of every significant design decision in this repo and *why* we made it, with pointers to the code that implements each one. Written to help a reader (hi Kinshuk) navigate the codebase quickly. This is the detailed companion to `PROJECT_UPDATE.md` (which is the current-state summary).

Last updated 2026-10-13.

---

## 0. The problem and the one-sentence approach
Wet-lab bacterial-colony experiments are **replicate-scarce**. The "augmenter" takes **one real replicate** of a condition and generates **new synthetic replicates of the same condition**, so one real plate becomes many training-quality plates. If the synthetic replicates are faithful *and* add genuine variation, downstream models trained in the scarce regime should improve.

**Core finding:** no newer generative backbone beats an in-domain SD1.5; the only thing that beats classical rotation downstream is **conditioning-dropout SD1.5** (smarter diversity on the same backbone), and it wins on the specific axis where its synthetic variation is faithful.

---

## 1. The augmenter model
- **Replicate-to-replicate ControlNet (source real → target real sibling of the same condition).** Rationale: we want the model to learn the *distribution of replicate-to-replicate variation*, not a fixed source→target mapping. Code: `reptorep_dataset.py`, `reptorep_finetune.py`.
- **Backbone = Stable Diffusion 1.5 + ControlNet, trained from the neutral `control_sd15_ini` seed (not warm-started from a primed ControlNet).** Rationale: a neutral init sits on the high-diversity side of the realism↔diversity tradeoff, which matters because downstream is diversity-gated (§4). Code: `branching_sd15_train_sa603.sh`, `ds_sd15_train_sa603.sh` (generic, any dataset).
- **256 px.** Rationale: native resolution of the branching/multiplexed data; higher res buys realism at a diversity cost (measured) and isn't the lever that matters for downstream.
- **Conditioning-dropout training (`HINT_DROPOUT=0.2`).** Rationale: randomly dropping the control signal during training forces the model to produce more varied-but-plausible replicates — more *useful* diversity at full inference strength, no realism cost. Code: `reptorep_dataset.py` (`HINT_DROPOUT` env), trained via `ds_sd15_train_sa603.sh ... HINT_DROPOUT=0.2`.

---

## 2. Evaluation metrics
- **`realism_vs_sibling` = LPIPS(gen → nearest real replicate of the same condition) ÷ the real-vs-real floor.** Lower = better; 1.0 ≈ as close to a real replicate as two real replicates are to each other. Rationale: normalizing by the irreducible replicate-to-replicate LPIPS makes the number interpretable and comparable across domains with different intrinsic variability.
- **CMMD (CLIP ViT-L/14 @ 336px), not FID.** Lower = better. Rationale: CMMD is reliable at the small sample sizes we have; FID is biased/unstable with hundreds of images.
- **`diversity` = mean pairwise LPIPS among a model's own samples for one source.** Higher = better. Rationale: a mode-collapse guard — a model can score well on realism while quietly collapsing.
- **`copy_rate`** checked and ~0 everywhere → no memorization, so novelty is not a confound.
- Code: `eval_metrics.py` (all image metrics; condition-matched by filename prefix via `--real_spec "TAG:/path"`), `cmmd.py` / `cmmd_direct.py` (direct CMMD for unconditional samples that `eval_metrics` can't condition-pair).

---

## 3. Scoring protocol and gotchas (read before re-running)
- **Score in a SEPARATE Slurm job from Flux generation.** Rationale: `eval_metrics.py` segfaults if a Flux pipeline was imported in the same process. Pattern: gen job → score job (`flux_scaled_gen_sa603.sh` then `flux_scaled_score_sa603.sh`).
- **Condition-matched generation naming.** Every generator writes `{condition}_{i}.png` (or `{source_stem}_s{i}.png`) so scoring/downstream can pair synth to the right real condition/source.
- **Always use the `youlab-gpu` partition.** Rationale: `gpu-common`/`scavenger-gpu` are Duke-wide and jobs sat PENDING for 13h; youlab-gpu has ~15 idle A5000 nodes and starts within ~1 min.
- **Heavy Python must run on compute nodes**, never the login node (a watchdog kills heavy login-node Python). Submit `sbatch` jobs or use `ssh dcc-agent`.

---

## 4. Downstream utility — the real test
Image realism is necessary but not sufficient; **downstream usefulness is the test that matters, and it's gated by diversity, not per-sample realism.**
- **The sharp experimental design:** classical **rotation is label-invariant** — a rotated colony has the same shape label, so it adds *zero* new information to a shape model. A **synthetic replicate has genuinely different morphology** with its own computed label, so it adds new signal — but only on the axis where the synthetic variation is faithful. This is why generative augmentation *can* beat rotation where rotation structurally cannot.
- **Protocol:** paired over seeds, data-scarce regime (K=1 real rep/condition), held-out **real** test set; arms = real / rotation / augmenter(in-domain) / augmenter(dropout) at matched image counts. The same seed selects the same K real sources across arms (true pairing). Code: `downstream_colony_features.py` (branching morphology), `downstream_multiplexed_decode.py` (multiplexed decode), and the generic `ds_downstream.py` (any decode/classification dataset; env-parameterized).
- **Labelers:** morphology features via `colony_features.py` (skimage two-stage segmentation → log-circularity "shape_factor", fractal dimension, solidity, eccentricity, edge_std; `--images` glob, domain-agnostic); decode labels parsed from filenames (aTc/IPTG for multiplexed, mixing ratio for KL, strain for Rattray).
- **Result:** conditioning-dropout SD1.5 beats rotation on branching `shape_factor` (+0.05, 15-seed, p≈0.05) and on multiplexed `aTc` decode (+0.045, 14/15 seeds); it ties on aggregate and can *lose* on axes where its variation isn't faithful (multiplexed `IPTG`). Axis-selective, not uniform.

---

## 5. Diversity levers (how we raise useful diversity without a new backbone)
- **Conditioning strength + guidance scale** (`sd15_divgen.py`, env `STRENGTH`/`SCALE`): lowering them reduces source-anchoring → more diversity, at a realism cost. Mapped the full frontier.
- **Conditioning-dropout training (the winner):** raises diversity even at *full* inference strength, so no realism cost. Sweet spot for branching = strength 1.0 / guidance 7.
- **Particle Guidance** (inference-only repulsion): fixed a high-dimensional bug (repulsion vanished as 1/h²); improves realism while raising diversity. `flux_gen_diverse.py`.
- **Stochastic DDIM eta:** a small free diversity boost (0.252→0.272) with no fidelity cost.
- **CADS:** null on this data.

---

## 6. Backbone search — what we tried and why
We tested the hypothesis "a newer/bigger backbone makes better replicates." It is wrong in this low-data regime. Every candidate and its rationale:
| Backbone | Why we tried it | Outcome |
|---|---|---|
| **SD1.5 in-domain** | the incumbent | best realism + CMMD; the one to beat |
| **Flux.1-dev + ControlNet** | newer flow-matching, 12B | escalated fp8→bf16, CN layers 2→6, 15k→20k steps; CMMD improved monotonically (8.8→4.8) but **realism plateaued at ~0.50**, never near SD1.5's 0.32. Gap is structural (faint/incomplete branches). Built the ControlNet from scratch (no released Flux ControlNet). Code: `flux_train_controlnet.py`, `flux_gen_eval.py`, `flux_{scaled,bf16}_*_sa603.sh`. |
| **PixCell** (cell-microscopy DiT, frozen prior) | domain-matched prior for cell-like colonies | **wins multiplexed image CMMD (5.78)** but washes out on branching (~50); never the downstream winner. The verdict flips by domain — a key finding. |
| **FastGAN / lightweight-gan** | literature says low-data GANs can beat diffusion at hundreds of images | unconditional, undertrained (16h timeout); CMMD 15.4 (branching) — lost. Not a usable augmenter (no source conditioning). |
| **StyleGAN2-ADA** | gold-standard low-data GAN | abandoned — custom CUDA ops wouldn't build with the available torch. |
| **SD3.5-medium + ControlNet** | flow-matching but NOT guidance-distilled → CFG works (the diversity lever that's dead on Flux) | abandoned — diffusers env dead-ends (dual_attention / accelerate). |
| **Zero-shot general priors (SDXL/SD1.5/SD3.5 img2img)** | a cheap screen before committing GPU-hours | catastrophic (CMMD ~71); for a domain this narrow, zero-shot breadth-screening does not discriminate — every untrained prior is ~equally useless. Lesson logged: "churn many candidates" must mean many *trained* candidates. Code: `screen_img2img.py`. |
| **From-scratch conditional DDPM** | no SD natural-image prior → test if that prior is the problem | learns structure but can't texture → blurry (realism 0.54, CMMD 9.8); 250-step re-sample confirmed it's structural, not undersampling. Code: `conditional_ddpm.py`. |

**Escalation discipline:** we only escalated Flux (the most promising alternative) and stopped when CMMD gains plateaued against a flat realism curve — spending more only to confirm the structural gap would not have been productive.

---

## 7. Dish standardization (presentation only)
- **Decision:** a **post-process**, not a retrain — detect each generated dish (Hough), rescale its rim to the measured real radius (117.9 px), composite only the colony interior inside a fixed real-glass template. Code: `dish_normalize.py`, `dish_composite.py`.
- **Dual-track rule (important):** composited images are for **figures/slides only**. Feeding them into downstream *hurts* it, because the fixed-radius feathering clips exactly the colony shape diversity that is the augmenter's edge. **Science uses raw synth; presentation uses composited.**

---

## 8. Wider-dataset validation (in progress)
Driven by Kinshuk's question: does the finding generalize beyond branching + multiplexed?
- **Dataset choices:** Rattray *P. aeruginosa* (62-way strain-ID; external, published, real replicate pairs), KL 2-species (mixing-ratio decode; cell-like), + the full backbone bake-off per dataset, + the sim-templated track. Rationale: span morphology / decode / regression / classification and internal / external / published.
- **Learnability gate before any generative training (`learnability_gate.py`):** train a ResNet on *real* images only; a dataset proceeds only if its downstream label is learnable above chance. Rationale: the single biggest waste-preventer — no ~40 GPU-hr bake-off on an unlearnable label. The real-only run also becomes the "real" arm of that dataset's downstream test.
- **Dropped swarmevo** despite a 99.8% gate pass: inspection showed the within-lineage images are **time-lapse frames of one plate**, not independent replicates → the 99.8% is memorization (same dish rim + handwritten label), the task is saturated, and the replicate paradigm doesn't apply. The naive "beats chance" check passed it; the frames-vs-replicates look caught it.
- **Core-finding-first ordering:** secure the dropout-vs-rotation result across all datasets (Phase 2) *before* the exhaustive per-dataset backbone bake-off (Phase 3), so the publishable answer lands early and the long tail can be stopped if needed.
- **Sim-templated kept as a separate track:** it's sim→exp, a different paradigm than replicate-to-replicate, so it must not contaminate the dropout-vs-rotation claim.
- Code: `gate_sa603.sh`, `ds_make_manifest.py`, `ds_sd15_train_sa603.sh`, `ds_gen.py`, `ds_downstream.py`, `ds_analyze.py`, `rattray_pilot_sa603.sh`.

---

## 9. Infrastructure & DCC conventions
- **Envs:** `flowmatch` (Flux/diffusers gen), `pytorch_PA_patternprediction` (SD1.5 ControlNet + scoring + downstream ResNets), `pixcell` (PixCell + lightweight image ops), `ddpm` (from-scratch DDPM, no opencv — use PIL), `lwgan` (FastGAN).
- **One generic script per job type, env-parameterized** (e.g. `ds_sd15_train_sa603.sh` takes `MANIFEST`/`FT_TAG`/`HINT_DROPOUT`) rather than a script per dataset — so adding a dataset is a new `--export`, not new code.
- **Big generated outputs are gitignored** (checkpoints, gen image dirs, `model_results/`, figures, decks); only code, small manifests, and result TSVs are committed. HTML viewers and slides stay local (regenerable).

---

## 10. Code map (where to look)
- **Train an augmenter:** `ds_sd15_train_sa603.sh` (generic) / `branching_sd15_train_sa603.sh` (the original) → `reptorep_finetune.py` + `reptorep_dataset.py`.
- **Generate synth:** `ds_gen.py` (generic, per-source) / `sd15_divgen.py` (diversity knobs) / `reptorep_infer_rattray.py`; Flux via `flux_gen_eval.py`.
- **Score image quality:** `eval_metrics.py`; unconditional via `cmmd_direct.py`.
- **Downstream utility:** `ds_downstream.py` (generic decode/cls) / `downstream_colony_features.py` (morphology) / `downstream_multiplexed_decode.py` (multiplexed); labeler `colony_features.py`; analysis `ds_analyze.py`.
- **Learnability gate:** `learnability_gate.py` + `gate_sa603.sh`.
- **Figures:** `make_backbone_matrix.py` (dataset×model matrix), `make_crossbackbone.py`, `dish_composite.py`.
- **Backbone bake-off jobs:** `mux_*_sa603.sh` (the per-backbone templates, reused across datasets by swapping `DATA_JSON`/`SOURCES`).
