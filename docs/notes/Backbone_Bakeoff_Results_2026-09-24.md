# Data augmenter backbone bake-off: results and recommendation
Sarthak, 2026-09-24

## Short version
We tested whether a newer or higher-resolution image backbone can push the replicate-to-replicate augmenter past its quality ceiling and improve downstream results. Across SD1.5 (current), PixCell-256, and PixCell-1024, no candidate cleanly beats SD1.5 for our production goal. The useful outcome is a clear map of the levers: downstream usefulness is gated by sample diversity, not per-sample realism, and diversity and realism trade off against each other. Light "from base" training raises diversity at a realism cost. Higher resolution raises realism at a diversity cost. SD1.5 already sits at a good point on that tradeoff, which is why it is hard to beat.

## What we tested
One frozen benchmark, identical scoring for every candidate (metrics v2: CMMD, realism-vs-sibling, diversity, copy_rate). Two domains:
- multiplexed aTc/IPTG dishes (has real downstream labels; native 256px)
- branching patterns ("Final_folder"; native 1001px, normally downsampled to 256)

Candidates:
- SD1.5 ControlNet (current), plus two cheap variants (decoder unlock, CADS sampler)
- PixCell-256, a microscopy-pretrained DiT backbone, native 256px
- PixCell-1024, same family at native 1024px, run on the branching data at its true resolution

Metric directions: realism-vs-sibling is lower-is-better (1.0 = as close to a real replicate as two real replicates are to each other). CMMD is lower-is-better (distribution match). diversity is higher-is-better (spread among a model's own samples for one input; near 0 means mode collapse). copy_rate is 0-is-best (no memorization). copy_rate was 0 everywhere, so novelty is not a concern for any candidate.

## Results

### Multiplexed (256px, the domain with downstream labels)
| model | realism-vs-sibling (lower better) | CMMD (lower better) | diversity (higher better) |
|---|---|---|---|
| SD1.5 (multiplexed fine-tune) | 0.412 | 12.6 | 0.381 |
| PixCell-256 | 0.346 | 5.8 | 0.259 |

PixCell-256 is better on realism (0.346 vs 0.412, good, images are closer to real replicates) and on CMMD (5.8 vs 12.6, good, the whole distribution is closer to real). But its diversity is worse (0.259 vs 0.381, bad, the samples look more alike). In the downstream decode, PixCell-256 did not beat the SD1.5 generalist. This is the core pattern of the whole project: better per-sample realism did not translate into better downstream, because downstream is limited by diversity.

### Branching (native 1001px): a clean recipe-vs-resolution ablation
All three arms were scored on the same held-out branching reals, so these are directly comparable.

| arm | realism-vs-sibling (lower better) | diversity (higher better) | CMMD (lower better) |
|---|---|---|---|
| PixCell-256, warm-start released cell ControlNet (full fine-tune) | 0.650 | 0.254 | 49.8 |
| PixCell-256, from base + small LoRA | 0.776 | 0.424 | 56.3 |
| PixCell-1024, from base + small LoRA (native resolution) | 0.599 | 0.342 | 55.1 |

SD1.5 reference on this domain: generalist best realism 0.559, CMMD 8.8.

Reading the single-variable comparisons (each row-to-row change moves exactly one factor):
- Change only the training recipe (warm-start to from-base) at fixed 256px: diversity rises a lot (0.254 to 0.424, good) but realism gets worse (0.650 to 0.776, bad). The from-base plus small-LoRA recipe is a diversity lever, paid for in realism.
- Change only the resolution (256 to 1024) at fixed from-base recipe: realism improves (0.776 to 0.599, good) and diversity drops (0.424 to 0.342, bad). Resolution is a realism lever, paid for in diversity.

The two levers push in opposite directions on the same tradeoff. PixCell-1024-from-base is the best joint point and beats the warm-start baseline on both axes, but it is one balanced point on a curve, not a break past the ceiling.

One branching-specific caveat: all PixCell arms have high CMMD (about 50 to 56) versus the SD1.5 generalist (8.8, much better). PixCell's prior is cell microscopy, and it does not fit dense branching morphology well even after fine-tuning. So on branching, PixCell is not the right backbone regardless of the tradeoff. On the cell-like multiplexed domain the opposite held: PixCell-256 beat SD1.5 on realism and CMMD.

## The central finding
Downstream usefulness is gated by diversity, not per-sample realism, and realism and diversity trade off. Two levers move that tradeoff:
- Training recipe: building the ControlNet from a neutral base and training a small LoRA gives more diverse samples than fully fine-tuning a strongly-primed ControlNet. More diversity, less realism.
- Resolution: higher native resolution gives more realistic samples. More realism, less diversity.

## Why SD1.5 is hard to beat, and recommendation
The current SD1.5 augmenter already trains its ControlNet from a neutral init (control_sd15_ini), so it already sits on the high-diversity side of the tradeoff (0.381 on multiplexed, the highest diversity we measured). A backbone swap does not obviously improve the thing that limits downstream. PixCell wins on cell-like domains for realism but not for downstream, and loses on branching.

Recommendation: keep SD1.5 as the production backbone. Treat resolution and training recipe as tunable knobs on the fidelity/diversity tradeoff, rather than expecting a backbone swap to break the ceiling. To push downstream further, the lever to pull is diversity, or a downstream task where orientation and fine detail matter more (there the augmenter's edge over classical rotation would show more clearly than on the rotation-friendly multiplexed dishes).

## Practical notes
- Native 1024 training and generation are feasible on the lab A5000s (fits in about 9 GB with LoRA, runs on youlab-gpu without queueing). Generation is slow unbatched (about 6 minutes per source). Batching the samples per source would give roughly a 4x speedup.
- Our eval sets are downsampled copies. The branching originals are natively 1001px and the multiplexed originals are 256 to 420px. For any future high-resolution work, pull from the originals in the Exp_images trove rather than upscaling small images.
- Optional future probe: LoRA versus full ControlNet on SD1.5, to check whether the diversity lever moves the production backbone. Expectation is small, since SD1.5 is already from-base and near the diversity ceiling.

## Methods and caveats
- Scoring is model-agnostic (eval_metrics.py). realism-vs-sibling is LPIPS to the nearest real replicate of the same condition, divided by the real-to-real floor. CMMD uses CLIP ViT-L/14 at 336px. diversity is mean LPIPS among a model's own samples for one input.
- Confound now controlled: the first branching comparison (256-warm-start vs 1024-from-base) changed both resolution and recipe. The 256-from-base control arm isolated them, which is how we attribute diversity to recipe and realism to resolution.
- The multiplexed PixCell-1024 arm was skipped. Multiplexed is natively 256px, so 1024 there is only upscaling, and the run timed out on the generation wall clock. It would not be a fair resolution test anyway.
- All copy_rate values are 0 (no memorization) across every candidate.
