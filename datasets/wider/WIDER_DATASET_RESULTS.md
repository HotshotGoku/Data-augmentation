# Wider-dataset validation — does the augmenter edge generalize?
Sarthak, 2026-10-15.

## The question
On our original two domains, conditioning-dropout SD1.5 beat plain rotation downstream on specific axes (branching `shape_factor`, multiplexed `aTc` decode). Kinshuk asked whether that holds more broadly. We tested it on two new datasets with real downstream tasks, plus a strengthening check and (in progress) a full backbone bake-off and the sim-templated track.

## Datasets & tasks
| Dataset | Source | Downstream task | Axis type |
|---|---|---|---|
| Branching | internal | morphology (`shape_factor`) | continuous morphology |
| Multiplexed | internal | aTc / IPTG decode | induction level |
| **Rattray 2023** | external, published (*P. aeruginosa*) | **62-way strain ID** | identity |
| **KL 2-species** | internal | **mixing-ratio decode** | continuous ratio |

## Method
- **Learnability gate first:** before any generative training, a ResNet on *real* images only confirmed each label is learnable (rattray 92% / 59× chance; KL R²=0.95). A fourth candidate, swarmevo, passed the naive gate at 99.8% but was **dropped** on inspection — its within-lineage images are time-lapse frames of one plate, not independent replicates (saturated, confounded).
- **Downstream protocol (identical to branching/mux):** paired over 15 seeds, data-scarce (K=1 real rep/condition), held-out **real** test; arms = real / rotation / augmenter(in-domain) / augmenter(dropout) at matched image counts.
- **Strengthening check:** after an initial negative, we retrained the augmenters longer (rattray 283 ep, KL 150 ep) to rule out an undertrained-augmenter artifact, and re-ran the 15-seed downstream.

## Result — the augmenter's edge is axis-specific, not general
Best augmenter config per dataset, paired Δ vs rotation (15 seeds). See `boundary_result.png`.

| Dataset — axis | Δ (aug − rotation) | t | verdict |
|---|---|---|---|
| Branching — shape_factor | **+0.052** | 2.14 | **win** |
| Multiplexed — aTc decode | **+0.045** | 6.8 | **win** |
| KL — mixing ratio | −0.001 | −0.06 | **tie** |
| Multiplexed — IPTG decode | −0.022 | −2.2 | lose |
| Rattray — strain ID | −0.108 | −10.1 | **lose** |

On both **new** datasets the augmenter does **not** beat rotation: KL ties (even after strengthening — the longer-trained augmenter went from −0.054 to −0.001), rattray loses decisively. The strengthening check rules out "the augmenter was just weak": more training helped but produced no win. The original branching/mux wins are real but **do not generalize as a blanket "beats rotation" claim**.

## Why — the augmenter regresses toward the mean, losing the extreme label value
The mechanism is the same on both new datasets and is visible directly in the synth (see `backbone_viewer/`):
- **Rattray:** a strain's real replicates share a fine rugose texture; the synth instead produce generic spiky/dendritic colonies — **realistic but strain-unfaithful**. With ~4 reps/strain the augmenter can't learn strain-specific variation, so it overrides the identity signal.
- **KL:** the mixing ratio is encoded in the red/green balance (1:19 = red-dominant, 19:1 = green-dominant); the synth of a 1:19 condition come out much greener — **the ratio is pulled toward the middle.**

So the rule is the original axis-selective claim, now with visual proof: **generative augmentation beats rotation only where its synthetic variation faithfully tracks the label; where it distorts the label, rotation wins** — because rotation is label-invariant and preserves the label exactly, which is a *strength* on identity/quantitative tasks and a *weakness* only on morphology axes (where it adds no new signal).

## Conclusion
This is a clean, honest **boundary result**: the "smarter diversity on SD1.5 beats rotation" finding holds on the axes where the augmenter is faithful (branching shape, mux aTc) and not elsewhere. That is more defensible and more publishable than an over-broad "it generalizes" — it maps *where* the method helps and gives a mechanism for *why*.

## Backbone bake-off — no backbone rescues it (2026-10-16)
We retrained two architecturally-distinct augmenter backbones on both new datasets and re-ran the 15-seed downstream vs rotation. **Neither beats rotation — both lose on both datasets**, confirming the negative is backbone-general, not an SD1.5 limitation. See `bakeoff_result.png`.

| Backbone | Rattray strain Δ | KL ratio Δ |
|---|---|---|
| SD1.5 (best config) | −0.108 | −0.001 (tie) |
| conditional DDPM | −0.196 | −0.058 |
| Flux ControlNet | −0.258 | −0.104 |

Mechanistic confirmation: the **less faithful** the backbone (Flux's blobby synth, DDPM's blurry) the **more** it hurts; the most faithful (SD1.5) hurts least — exactly what the mean-regression / label-unfaithfulness mechanism predicts. (FastGAN is unconditional, so it has no per-source augmenter arm; PixCell omitted since two trained backbones already confirm the pattern.)

## Still in progress
- **Sim-templated track** (Kinshuk's sim→exp domain; real vs real+synth) — a *different* paradigm than replicate-to-replicate, reported separately; downstream task being scoped.

## Artifacts
Figure `boundary_result.png` (builder `make_boundary_fig.py`); data `{rattray,kl}_downstream{,_15,_long}.tsv`; harnesses `ds_downstream.py`, `ds_gen.py`, `learnability_gate.py` (see `METHODOLOGY.md`).
