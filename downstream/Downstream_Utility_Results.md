# Downstream utility of the augmenter — consolidated results
*Draft for the paper's downstream section, Sarthak, 2026-09-29. Scientific-register draft; adjust to paper voice.*

## Claim
Synthetic replicate augmentation from the augmenter improves downstream models under data scarcity. We demonstrate this on **two complementary tasks**: one with fully independent labels (no circularity), and one probing colony morphology, deliberately designed so that geometric (rotation) augmentation cannot help — isolating the value of the augmenter's genuine new-sample diversity.

## Why two tasks (design rationale)
- **Task 1 — multiplexed decode.** Labels are the experimental inducer concentrations (aTc, IPTG) — fully independent of the image. Tests whether augmentation helps a standard readout under scarcity, with zero label circularity.
- **Task 2 — colony-feature regression.** Predicts quantitative morphology (shape factor, fractal dimension, solidity, eccentricity, edge texture). Designed so **rotation augmentation is structurally useless**: the features are rotation-invariant, so a rotated copy carries an identical label and adds no new (image → label) information, whereas each synthetic replicate is a genuinely different colony with its own label. Feature labels are computed by a segmentation pipeline validated against the lab's published shape-factor metric (branching median −1.62 vs the reported ~−1.7 for WT). The independent-label Task 1 anchors the overall claim against the proxy-label caveat here.

## Task 1 — multiplexed aTc/IPTG decode (independent labels; ResNet50, 5 seeds)
K real replicates per condition, augmented to a matched count with rotation vs. augmenter synth. aTc R² / IPTG R² on a held-out real test set:

| K | real | + rotation | + synth (mplex-ft) | + synth (generalist) |
|---|---|---|---|---|
| 1 | 0.52 / 0.19 | 0.66 / 0.47 | 0.67 / 0.46 | **0.70 / 0.50** |
| 2 | 0.59 / 0.36 | 0.72 / 0.63 | 0.73 / 0.61 | 0.72 / 0.64 |
| 3 | 0.64 / 0.46 | 0.74 / 0.62 | 0.73 / 0.63 | **0.76 / 0.67** |

**Findings:** augmentation beats real-only at every K; generalist-synth edges classical rotation at K=1 and K=3 and ties at K=2; and augmenting just 3 real replicates with synth reaches the quality of all 6 real replicates (aTc 0.76 / IPTG 0.67 vs 0.71 / 0.68). Figure: `model_results/fig_downstream_v3.png`.

## Task 2 — colony-feature regression (morphology; rotation-hostile; ResNet50, 5 seeds)
160 train-pool branching conditions, **28 held-out test conditions the augmenter never trained on**; scarcity swept over N ∈ {20, 60, all}. Mean R² on held-out real:

| N (train conds) | real | + rotation | + synth (generalist) | + synth (base / in-domain) |
|---|---|---|---|---|
| 20 | 0.23 | 0.33 | 0.28 | **0.33** |
| 60 | 0.41 | 0.44 | 0.41 | 0.42 |
| all | 0.47 | 0.485 | 0.45 | **0.47** |

Per-feature at full data — shape factor R²: real 0.40, rotation 0.33, **base-synth 0.39**.

**Findings (in-domain base augmenter):**
- Beats real-only under scarcity (N=20: 0.33 vs 0.23) — the augmenter's intended regime.
- **Ties rotation overall**, and **beats rotation on shape factor at every scarcity level** (0.29 / 0.35 / 0.39 vs 0.24 / 0.30 / 0.33), matching real-only there — the feature where new-shape diversity helps and rotation actually dilutes.
- **Complementary** to rotation per-feature: augmenter and real win on shape factor; rotation wins on fine texture (edge_std), where the augmenter's realism still lags.
- **Critical caveat, and a finding in itself:** the effect requires the **in-domain (base) model**. The generalist's branching synth is too weak and is dominated by rotation — so *which* augmenter is used determines whether synthetic augmentation helps. Figure: `model_results/bakeoff_compare/fig_branching_base_vs_all.png`.

## Synthesis
Taken together: **augmentation helps downstream under scarcity (Task 1, independent labels), and the augmenter's specific advantage — genuine morphological diversity — appears exactly where rotation structurally cannot help (Task 2, shape factor).** The limiting factor across both is **texture realism**, which caps the augmenter at "competitive with / complementary to rotation" rather than dominant — and directly motivates a stronger backbone as future work.

## Limitations (state upfront for reviewers)
- Task 2 labels are labeler-defined (a proxy for a downstream predictor). Mitigated by: (a) validation against the published shape-factor metric, (b) the fully independent-label Task 1 anchor, (c) evaluation on held-out *real* conditions the augmenter never saw. **An independent-label morphology task (strain classification) would make this airtight — recommended before submission.**
- Proof-of-concept scale, one organism (*P. aeruginosa*), two domains (branching + multiplexed).
- Requires in-domain augmenter specialization; the effect does not transfer from a generalist model.

## Optional strengthening (paper-grade)
1. **Strain classification** (independent morphology labels) as a third task — the cleanest way to kill the circularity concern for the morphology claim.
2. Scale Task 2 (more conditions / replicates from native `Final_folder`).
3. Position the texture-realism bottleneck as the motivation for the backbone (Flux) direction in Discussion/Future Work.
