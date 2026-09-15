# Replicate Augmenter — Project Update (September 2026)

Companion writeup to `Augmenter_Update_Sept2026.pptx`. For Kinshuk.

## Goal
Wet-lab bacterial-colony experiments are replicate-scarce. The augmenter takes **one real replicate**
and generates **new, realistic replicates of the same condition** (ControlNet + Stable Diffusion 1.5,
replicate-to-replicate). If the synthetic replicates are faithful, one real plate becomes many
training-quality plates — beating data scarcity for downstream models.

**Metric.** `realism_vs_sibling` = LPIPS from a generated image to real replicates of the same
condition, reported as a **ratio to the real-vs-real floor** (1.0× = indistinguishable from a real
replicate; higher = worse). Diversity (mode-collapse guard) and FID reported alongside.

## The problem
- **In-distribution:** faithful (~1.3×, near ideal).
- **Out-of-distribution:** fails — ignores the input and paints its trained pattern (2.5–3.0×).
Closing that gap is the project.

## Results

**1. Fine-tuning is the lever — confirmed on 3 domains.** A few epochs on target data:
- 2-species **2.54× → 1.16×**, multiplexed **2.96× → 1.52×** (FID 314→105), Rattray **2.18× → 1.69×**.
- Inference-only knobs (guidance scale, control strength) do **not** help; fine-tuning is required.
- Visual: only the multiplexed-fine-tuned model reproduces the sparse red-dot pattern (see deck slide 5).

**2. A generalist model gives positive cross-domain transfer.** One model jointly fine-tuned on all
internal domains (5,722 replicate pairs):
- Beats the base on **4/6 domains** (kl2 2.88→1.76×, ff 2.71→2.21×, selx 2.72→2.30×, mplex 2.96→2.30×).
- Improves FID ~2× and **fixes mode-collapse on all 6** (diversity 0.14→0.26; the base was quietly collapsed on branching).
- Distant off-target data (SwarmEvo) did **not** help — on-target-ness matters, not raw volume.
- Trade-off: a **specialist still wins per target** (dedicated multiplexed 1.52× vs generalist 2.30×). The generalist is best as a universal default / warm-start base.

**3. Data scarcity is solved — internally.** `ks723/storage/Exp_images` holds **399 GB** of on-target
colony replicate data on the cluster (no outreach). Catalogued **~4,500 replicate pairs**:
EmrahPaKp (P. aeruginosa + K. pneumoniae, 1,988), KL 2-species (1,096), Final_folder branching (514),
NL_evolution (504), Selected_Exps (374), + multiplexed (1,932).

**4. Downstream utility — the honest result.** Correspondence-safe test: decode the two multiplexed
inputs (Row 10-way + Column 7-way) from the pattern with a ResNet18, in a data-scarce regime, comparing
real-only vs classical rotation vs augmenter-synth (matched image counts, 3 seeds, held-out real test).
- Augmenter **wins only at K=1** (one real replicate): joint acc 0.133 vs 0.101 real vs 0.117 classical — ~2× the classical gain from a single replicate.
- At **K≥2 the differences are within seed-noise**; classical rotation trends best.
- This is the **second** downstream test without a clear win beyond extreme scarcity. **Realism ≠ utility**, and usefulness is gated by how good the augmenter is (1.52× on this domain isn't enough).

**5. The ceiling is generative, not the VAE (flow-matching de-risked).** Before a multi-day SD3.5 build:
- **Warm-start** (specialize on multiplexed *from the generalist* vs *from base*, same recipe) → both hit
  **1.52×**: initialization doesn't matter, so it's a generation limit.
- **VAE reconstruction diagnostic:** even SD1.5's 4-channel VAE reconstructs multiplexed at **LPIPS 0.02 —
  ~7× below the 0.14 realism gap**. A 16-channel VAE (SD3.5's headline advantage) improves recon by ~0.01,
  negligible. **The VAE is not the bottleneck.**
- → the 1.52× ceiling is in the **generation** (source→target mapping), not representation. SD3.5's VAE
  wouldn't help. Whether its flow-matching *denoiser* would has no cheap test. The diagnostic **saved the build.**
- Nuance: for stochastic sparse-dot patterns, per-sibling LPIPS penalizes valid-but-different random
  realizations; FID improved a lot (314→105), so 1.52× likely overstates the real problem here.

## Bottom line
The augmenter reliably **adapts** to any domain and makes realistic, diverse replicates; **data is no
longer the blocker**. Its **downstream value today is real but confined to extreme scarcity (K=1)**, and
pushing realism higher on SD1.5 is capped by the generator (not the VAE or data).

## What would help from you
1. **Condition metadata** for a strong-augmenter domain (e.g. 2-species, where the augmenter is 1.16×) so we can test utility where it's near-perfect — the fairest test we haven't been able to run (labels are opaque without it).
2. Your read: is **distributional realism (FID)** good enough for the intended use, or is per-replicate fidelity essential?
3. **Priorities** among the internal datasets — which experiments matter most to augment.
4. **Go / no-go** on a full SD3.5 build to test the flow-matching *generator* (multi-day, uncertain payoff).

## Key artifacts (DCC: `/hpc/group/youlab/sa603/code/Data_augmentation/`)
- Base: `lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt`
- Specialists: 2sp `finetune_runs/shallowsweep_lr5e-6/.../epoch=3`, Rattray `rattray_runs/rat_epochcurve/.../epoch=3`, multiplexed `rattray_runs/multiplexed_ft/.../epoch=2-step=1448.ckpt`
- Generalist: `rattray_runs/generalist/.../epoch=3-step=5723.ckpt`
- Data: `data/multiplexed_eval/`, `data/generalist/` (canon + per-domain test), source trove `ks723/storage/Exp_images/`
- Utility experiment: `downstream_multiplexed_out/results.tsv`; figures in `model_results/`
- Flow-matching env (kept): `envs/flowmatch`; VAE diagnostic: `reconstruct_vae_test.py`
