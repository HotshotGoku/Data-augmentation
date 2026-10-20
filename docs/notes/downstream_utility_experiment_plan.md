# Downstream-Utility Experiment — does synthetic augmentation improve pattern prediction?

*Draft plan for Sarthak + Kinshuk · Aug 2026*

## Goal
Prove the augmenter's synthetic replicates **improve a downstream pattern-prediction model** — the open deliverable in the project profile. Show it especially under **data scarcity** (the augmenter's whole purpose: beat experimental data scarcity).

## What the sim project offers (from repo investigation)
- **sim→exp ControlNet** (Fig 5, the lab's flagship generative predictor): simulation image → experimental colony. **Exp images are the TARGETS.** Trained on `prompt_simtoexp.json` (~409 base sim/exp pairs × ~100 rotations = 40,900). Held-out **`Exp_testset` (96 real)**; metrics SSIM / LPIPS / ORB in `sim_to_exp_diffusion/controlnet_essential/cldm/metrics.py`. → synthetic replicates plug in as extra targets.
- **exp→seed UNet** (Fig 6, inverse): experimental colony → seed. **Exp images are the INPUTS.** Metric = seed reconstruction. → synthetic replicates plug in as extra inputs (less circular).
- **dilResNet seed→sim/sim→sim** (Figs 2–4): simulation-space only; VAE-latent targets. Not a fit for an *experimental*-image augmenter.

## Two honest notes that shape the design
1. **Domain:** the downstream infra is in-domain **branching**, so we use the **base augmenter** (in-domain), not the 2-species fine-tuned one. The 2-species work proved the *fine-tuning method*; this proves *utility* on the lab's existing pipeline.
2. **What kind of augmentation:** the augmenter is replicate-to-replicate — it generates more replicates of **existing** colonies (intra-colony diversity), not brand-new colonies. So the claim we test is "**generative replicate-augmentation improves the predictor**," which should matter most when real data is scarce.

## Recommended design — sim→exp, data-scarcity curve
Downstream = sim→exp ControlNet. Vary real-data size N ∈ {50, 100, 409 (all)} base colonies:
- **A (real-only):** train sim→exp on N colonies (× rotation aug, as the paper does).
- **B (real + synthetic):** same N colonies **plus** K synthetic replicates/colony from the augmenter, each paired with that colony's simulation.
- **Evaluate every model on the SAME held-out real `Exp_testset`** (SSIM/LPIPS/ORB).
- **Hypothesis:** B > A at each N; the gap is largest at small N; ideally **B at small N ≈ A at large N** (synthetic substitutes for real data) — the money result.

## Phases
1. **Generate synthetic replicates** — run the base augmenter (`pipeline.py`, `CKPT_PATH_V4`) on the sim→exp `Exp/` colonies → K replicates each. **Reconcile preprocessing**: the augmenter emits `backgroundblack`-style images while sim→exp targets use `preprocess_experimental_backgroundwhite` — must match the target format. (Reuse `reptorep_infer` machinery.)
2. **Build per-condition training JSONs** — subset `prompt_simtoexp.json` to N colonies (A); add `{sim → synthetic_exp}` lines (B).
3. **Train** sim→exp per (N × condition), resume from `control_sd15_ini.ckpt`, ~5 epochs (mirror `simtoexp_train.py`; `--exclusive`; score-then-prune). Compute ≈ a few hrs each × ~6 runs ≈ 1–2 GPU-days.
4. **Evaluate** each on `Exp_testset` with `cldm/metrics.py` (SSIM/LPIPS/ORB) → summary table + a performance-vs-N curve (A vs B).
5. **Document** — manifests + tracking + example-image montages (sim → real exp → A-predicted → B-predicted), committed.

## Minimal first version (fast sanity)
One N (e.g., 100) × {real-only, real+synthetic} → 2 trainings + eval. Confirms the effect exists before running the full curve.

## Alternatives
- **exp→seed UNet** — less circular (different architecture, different target); synthetic exp as extra *inputs*; metric = seed reconstruction. Good robustness check / second experiment.
- **Build a 2-species downstream task** — a new predictor/classifier + test set for the 2-species domain, to showcase the *fine-tuned* augmenter. More novel but requires building infra from scratch.

## Risks
- **Format mismatch** (backgroundblack vs backgroundwhite) — reconcile in Phase 1.
- **Circularity** (augmenter & sim→exp are both diffusion on exp images) — mitigate via real held-out test + the scarcity framing; the exp→seed alternative removes it.
- **Compute + disk** (multiple trainings) — sequential + score-then-prune.

## Verification
Curve/table shows B ≥ A on the real held-out `Exp_testset` (esp. at small N), plus montages that visually show B's predictions closer to real than A's.
