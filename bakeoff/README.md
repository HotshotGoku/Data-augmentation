# bakeoff/ — backbone search (CLOSED)
Question: does any newer generative backbone beat an in-domain SD1.5 replicate augmenter?
**Answer: no.** Verdict + metrics: `docs/PROJECT_UPDATE.md`. One subfolder per backbone tried:
`flux/` `pixcell/` `sd3_sdxl/` `ddpm/` (from-scratch) `gan/` (FastGAN/StyleGAN2-ADA) `bbdm/`
`sd15/` (incumbent re-test) `screening/` (zero-shot screen). `_shared/` = learnability/sim gates,
cross-backbone matrix + comparison figures.
