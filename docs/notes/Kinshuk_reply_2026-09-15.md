# Reply to Kinshuk (draft, 2026-09-15)

Thanks! Answers below, and I updated the branch + wrote a definitions doc so this is all easier to follow.

## 1. How I build datasets, and what "100 pairs" means

For each dataset I canonicalize the images to 256px, then build replicate pairs within each condition or strain. A "pair" is one real replicate as the source and a different real replicate of the same condition as the target, written one per line to a JSON. Fine-tuning reads that JSON. I keep whole conditions or strains out of the pair-building, so the test set has zero leakage.

When I say "100 pairs," I mean 100 raw replicate pairs, that is 100 lines in the fine-tuning JSON. It is not 100x augmented. I do still augment, but on the fly: each pair gets a random 90 degree rotation and a random flip every time it is loaded, so over 8 epochs each pair is seen about 8 times in different orientations. Two differences from your setup worth flagging:
- You physically pre-generate about 100 rotated copies per image, so your dataset on disk is 100x larger. Mine stays at 100 and augments stochastically at load time. Same rotation-invariance idea, different bookkeeping.
- My 100 are 100 genuine replicate-to-replicate pairs (real biological siblings), not one image rotated into many, so the model is learning real sibling variation with rotation and flip on top.

Why it adapts on so few: the base is already a strong colony prior, so this is adapting an existing model, not training from scratch, and each pair carries real sibling variation, which is exactly what we want it to learn.

Two honest caveats so you can calibrate:
- On data-efficiency (now with error bars over 3 seeds): base-model quality improves steadily and flattens by about 250 pairs. 100 pairs already captures most of the gain (realism-vs-sibling about 0.63, versus 0.58 at the full 720 and 0.69 at only 10; the real-vs-real floor, which would be ideal, is 0.32). So the honest statement is that a small amount of target data gets you most of the way, not that 100 pairs is perfect.
- On your point that smaller N also trained for fewer steps: I checked this directly. I re-ran the curve at the early optimum with error bars, and separately ran a convergence check (fine-tuning a small N=25 and the full N=720 while saving every epoch). Both plateau within about one epoch, so adaptation is epoch-driven and small N is not undertrained. The gap between small-N and full-N quality is therefore a data effect, not a fewer-steps artifact. I can also run it with your exact 100x rotation pre-augmentation if you want that as an extra check.

## 2. Base and generalist definitions, with the numbers

Your base definition is right. Your generalist definition is off in two ways, so let me correct it.

**Base**: the model I trained at the start on your 3 datasets (already 100x rotation-augmented):
- EmrahPaKp (the PaKp set): 198,800 pairs
- Final_folder (the branching set, your paper): 51,400 pairs
- NL_evolution (the evolution / mutant library): 50,400 pairs
- Total 300,600 pairs, 5 epochs. Can you confirm the person-attributions on these three so I name them correctly in the doc?

**Generalist**: fine-tuned from the base, but not on "Kristen + Rattray + multiplexed." It is jointly fine-tuned on 6 on-target domains, 5,722 raw pairs:

| domain | train images | train pairs | held-out test (sources / reals) |
|---|---|---|---|
| pakp (EmrahPaKp) | 486 | 1,762 | 18 / 68 |
| kl2 (Kristen 2sp) | 184 | 860 | 5 / 36 |
| ff (Final_folder) | 402 | 410 | 28 / 64 |
| nlev (NL_evolution) | 252 | 432 | 12 / 36 |
| selx (Selected_Exps) | 358 | 326 | 24 / 48 |
| mplex (multiplexed) | 403 | 1,932 | 70 / 417 |
| rattray (held out from generalist) | 248 | 720 | 13 / 48 |

"train images" = unique real images used to build pairs; a "pair" is a source-to-target combination
of two replicates of the same condition, so pairs exceed images when a dataset is replicate-rich
(pakp) and can be fewer when it is replicate-sparse (selx). The base's 3 datasets above are reported
as your pre-augmented (AUG100) pair counts, so their raw image counts are not directly comparable; the
unique-image scale for the same underlying sets is visible in the generalist rows (pakp 486, ff 402,
nlev 252).

Two things versus your assumption. Rattray is not in the generalist. I deliberately held it out so it is unseen by both the base and the generalist, which makes it my clean few-shot testbed for the adaptation curve. And the generalist re-includes your 3 base datasets (as canonical replicate pairs) and adds Kristen, Selected_Exps, and multiplexed. So the generalist is everything on-target I could canonicalize into clean pairs, except Rattray.

And yes, the CMMD graph evaluates on held-out test sets per domain. I hold out about 15% of conditions per domain (every 7th condition) from training and score CMMD-336 on those. For Rattray it is 13 held-out strains with zero overlap with the 64 training strains.

## 3. Downstream R2, your rigorous checks, and definitions

You were right that R2 was on the low side, and it was mostly the decoder. I re-ran with a ResNet50 (was ResNet18), 80 epochs, and model selection done separately for the classification and regression heads. That lifted R2 a lot and, interestingly, tipped the augmenter ahead of classical rotation. Mean over 5 seeds (aTc R2 / IPTG R2):
- K=1: real 0.52 / 0.19, classical 0.66 / 0.47, augmenter-mplexft 0.67 / 0.46, augmenter-generalist 0.70 / 0.50
- K=2: real 0.59 / 0.36, classical 0.72 / 0.63, augmenter-mplexft 0.73 / 0.61, augmenter-generalist 0.72 / 0.64
- K=3: real 0.65 / 0.46, classical 0.74 / 0.62, augmenter-mplexft 0.73 / 0.63, augmenter-generalist 0.76 / 0.67
- reference, all 6 real replicates (no augmentation): aTc 0.71, IPTG 0.68

Takeaways: augmentation beats real-only at every K for both readouts (for example K=1 IPTG goes from 0.19 real-only to about 0.47 to 0.50 with augmentation). With the stronger decoder the generalist-synth augmenter now edges out classical rotation at K=1 and K=3 and ties it at K=2, where before (ResNet18) classical was matching or beating it. And notably, augmenting just 3 real replicates with generalist synth (aTc 0.76 / IPTG 0.67) reaches the quality of using all 6 real replicates (0.71 / 0.68), so the synthetic data buys back most of the missing real data. The generalist synth also beats the dedicated multiplexed-ft synth. Figure: fig_downstream_v3.png (all points now 5 seeds).

For your own checks, the branch is updated and here are the checkpoint files (all readable by you, world-readable under youlab, no copy needed):
- base: `.../lightning_logs/version_52034860/checkpoints/epoch=4-step=375749.ckpt` (8.1G)
- 2sp-ft: `.../finetune_runs/shallowsweep_lr5e-6/lightning_logs/version_53408286/checkpoints/epoch=3-step=75999.ckpt` (7.7G)
- rattray-ft: `.../rattray_runs/rat_epochcurve/lightning_logs/version_54107493/checkpoints/epoch=3-step=719.ckpt` (8.1G)
- multiplexed-ft: `.../rattray_runs/multiplexed_ft/lightning_logs/version_54549115/checkpoints/epoch=2-step=1448.ckpt` (8.1G)
- generalist: `.../rattray_runs/generalist/lightning_logs/version_54592625/checkpoints/epoch=3-step=5723.ckpt` (8.1G)

(All under `/hpc/group/youlab/sa603/code/Data_augmentation/`.)

I also generated ready-to-use synthetic replicate folders at `/hpc/group/youlab/sa603/data/multiplexed_eval/synth_ft` and `.../synth_generalist` (about 1,680 images each, named `R{row}C{col}_src{rep}_s{n}.png`, with a `SYNTH_README.md` alongside), from the multiplexed-ft and generalist models, plus the generation script (`gen_multiplexed_synth.py`) so you can regenerate or extend. And I wrote a definitions doc on the branch (`MODELS_AND_DATASETS.md`) covering base versus generalist, the exact training composition, what a pair and N pairs mean, the metrics, and the checkpoint inventory.

One question back: which `labels_aTc_IPTG*.json` is authoritative for the Combined train, validation, and test splits? I found several variants and want the downstream ground truth to use the right one.

## Quick questions for you
1. Confirm the person-attributions for the base's 3 datasets (your paper / Jia's PA mutant library / PaKp mapping to Final_folder / NL_evolution / EmrahPaKp).
2. Which `labels_aTc_IPTG*.json` is authoritative for the Combined splits?
3. OK with the nearest-neighbor per-sample framing (each generated image compared to its most similar real replicate, read against the real-to-real floor)?
