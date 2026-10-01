# Literature survey — colony features + generative augmentation prior art
Sarthak, 2026-09-28 · for Kinshuk (point 6: agentic paper search for colony features + similar bioRxiv/arXiv work)

## How this was produced (the agentic-search angle Kinshuk raised)
Ran two parallel agentic web sweeps (live search + fetch), mirroring the Reker-lab "source discovery → structured extraction" pattern from the seminar slides — one on **colony-feature quantification**, one on **generative-augmentation prior art**. Below is the curated result; every source is linked. This scales: the same harness could be pointed at a standing bioRxiv/arXiv watch for our topic. Two items the sweep couldn't fully fetch are flagged "verify."

**Heads-up:** the prior-art sweep surfaced a bioRxiv paper that is **our own lab's** — Sahu, *Simulation-templated photorealistic prediction of bacterial patterns* (Dec 2025), which matches the `Simulation_templated_pattern_prediction-main` repo in this project. So it's the sibling simulation work, not a competitor; this augmenter is its replicate-augmentation extension.

---

## Part A — Colony-feature quantification (targets for a downstream task)

The point of this half: pick concrete, auto-labelable colony features for a downstream task where **fine structure/orientation matters** — the regime where a generative augmenter should beat plain rotation/flip. All targets below are computable with open tools (no manual annotation).

Most relevant sources:
- **Luo, … You. *Mol Syst Biol* 2021 — branching optimizes P. aeruginosa colony growth.** [EMBO](https://www.embopress.org/doi/full/10.15252/msb.202010089). Defines **shape factor = log₁₀(circularity)** (round↔branched; WT ≈ −1.7) + area, equivalent-disk radius, perimeter, mean centroid distance, eccentricity, caliper diameter. **This is a You-lab paper — our own prior art and metric vocabulary.**
- **Rattray et al. *PLoS Comput Biol* 2023 — ML strain-ID from P. aeruginosa colony images.** [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011699) · [code](https://github.com/GaTechBrownLab/Rattray-2023-PLOSCompBio). 8 morphology + 7 Sobel-edge-gradient "complexity" features; **already an external dataset in this project.** Best template for a colony-feature regression head, with code.
- **FracLac (Fiji)** [manual](https://imagej.net/ij/plugins/fraclac/fraclac-manual.pdf) — box-counting **fractal dimension + lacunarity** on masks; the go-to for auto-generating fractal/texture labels. Foundational method: **Obert et al. *J Bacteriol* 1990** (mass vs boundary fractal D) [DOI](https://doi.org/10.1128/jb.172.3.1180-1185.1990). B. subtilis radially-resolved fractal precedent: [ESPR 2022](https://link.springer.com/article/10.1007/s11356-022-19817-4).
- **BiofilmColony-morphometrics** [repo](https://github.com/JBlackburnCode/BiofilmColony-morphometrics) — turnkey Python (OpenCV/scikit-image) for plate photos: area, Crofton perimeter, circularity, solidity, equivalent diameter, GLCM contrast/entropy. Closest ready-made extractor to bolt on.
- **Two-species mixing:** **Dogsa & Mandic-Mulec, *Biofilm* 2023** [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10542597/) · [Fiji code](https://github.com/IztokD/MSSegregation-package) — multiscale spatial segregation (rMSSL); and **Natan et al. *Sci Rep* 2022** [paper](https://www.nature.com/articles/s41598-022-20644-3) (cross-species spatial Pearson correlation). For the fluorescence co-cultures.
- **Standard software / datasets:** BiofilmQ (*Nat Microbiol* 2021) [paper](https://www.nature.com/articles/s41564-020-00817-4); AnaMorf (branch length/tortuosity) [repo](https://github.com/djpbarry/AnaMorf); AGAR colony dataset (*Sci Data* 2023) [paper](https://www.nature.com/articles/s41597-023-02404-8); high-res clinical colony set (*Sci Data* 2026, Zenodo) [paper](https://www.nature.com/articles/s41597-026-07095-5).

**Recommended feature targets** (standard + fine-structure-sensitive + should benefit from generative replicates under scarcity):
1. **Shape factor = log₁₀(circularity)** — canonical round↔branched axis (Luo/You 2021); perimeter-driven, reacts sharply to branch fineness. *Primary target.*
2. **Box-counting fractal dimension** (mass + boundary) — ramification/space-filling; rotation-invariant, so rotation aug can't enrich it — generative diversity can.
3. **Lacunarity** (same FracLac pass) — distinguishes patterns with equal fractal D but different branch spacing.
4. **Edge-gradient complexity stats** (SD/skew/kurtosis of Sobel) — proven to separate P. aeruginosa strains (Rattray 2023, code available).
5. **Solidity + convex-hull ramification** (optionally branch length/tortuosity, AnaMorf) — lobing/branch reach.
6. **Multiscale segregation (rMSSL) / cross-species correlation** — the two-species task; intrinsically about fine spatial arrangement (clearest "generative should win" case).

Split: 1–5 for the P. aeruginosa branching task; 6 for the fluorescence co-culture. Targets 2–4 are the strongest "generative-augmentation-should-win" bets — they depend on high-frequency structure that rotation/flip leaves statistically unchanged.

---

## Part B — Generative augmentation prior art (positioning + what to borrow)

Closest bacterial competitors:
- **Holicheva et al. *npj Biofilms & Microbiomes* 2025 — generative annotated biofilm images.** [paper](https://www.nature.com/articles/s41522-025-00647-4). VAE/WGAN/diffusion/CycleGAN pipeline; Mask R-CNN downstream; pattern-aware synthetic matched **70 real images**. Composites single cells, not whole-colony replicates — the nearest published bacterial competitor.
- **Pawłowski et al. *Sci Rep* 2022 — colony synthesis by style transfer.** [arXiv](https://arxiv.org/abs/2111.03789). From **100 real images**, synthetic-trained detector mAP 0.416 vs 0.520 real. Canonical whole-plate colony synthesis under scarcity.
- **Sahu, bioRxiv 2025 — simulation-templated bacterial patterns** [link](https://www.biorxiv.org/content/10.64898/2025.12.13.694038v1) — **our own lab's sibling work** (verify version/metrics on a full read).

Microscopy augmenters with measured downstream gains (borrow the eval designs):
- **Nudiff (MICCAI 2023)** [arXiv](https://arxiv.org/abs/2310.14197) — paired mask→image diffusion; **10% real + synthetic = full baseline** on nuclei segmentation.
- **Eschweiler et al. *PLoS Comput Biol* 2024** [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10906858/) — sketch-conditioned DDPM (ControlNet-like); **70–80% of real replaceable** by synthetic, no accuracy loss.
- **SynCellFactory (2024)** [arXiv](https://arxiv.org/abs/2404.16421) — **ControlNet** cell-video augmenter, helps most when data is sparse (near-sibling architecture).
- **EMIT-Diff/DiffBoost (2023)** [arXiv](https://arxiv.org/abs/2310.12868) — edge-conditioned diffusion; +7–14% Dice on medical segmentation.
- **HistoGAN selective augmentation (2021)** [pdf](https://arxiv.org/pdf/2111.06399) — **naive synthetic can hurt; gains need quality-gated selection** (matches our sim→exp negative — curate, don't dump).

Backbone / method templates most aligned with us:
- **Augmented Conditioning (2025)** [arXiv](https://arxiv.org/html/2502.04475) — freeze a diffusion model, condition on **an augmented real source image** → in-domain variants; **augmentation strength is an explicit realism↔diversity dial**, up to +25% few-shot. **The cleanest published mechanism for the exact tradeoff we measured — top thing to borrow/cite.**
- **PixCell (2025)** [arXiv](https://arxiv.org/abs/2506.05127) · [code](https://github.com/cvlab-stonybrook/PixCell) — the DiT we bake-off-tested; conditions on self-supervised image embeddings (the "embed a source, generate variants" motif).
- **MorphoDiff (ICLR 2025)** [biorxiv](https://www.biorxiv.org/content/10.1101/2024.12.19.629451v1) — SD-latent cell-morphology prior, finetunable.

Realism-vs-diversity, and the "better backbone" hypothesis (Kinshuk's point 1):
- **CellFlux (ICML 2025)** [pdf](https://arxiv.org/pdf/2502.09775) — **flow matching beat diffusion by +35% FID *and* +12% mode-of-action accuracy simultaneously** in cell morphology. The single strongest data point that a stronger paradigm improves fidelity *and* faithfulness at once — direct support for the Flux/flow-matching bet.
- **DiADM (2024)** [arXiv](https://arxiv.org/html/2411.16171v2) — fidelity and diversity are architecturally **decouplable** (improve diversity with no quality loss). SOTA diffusion captures <77% of dataset diversity.
- **"When Pretty Isn't Useful" (2026)** [pdf](https://arxiv.org/pdf/2602.19946) *(verify — PDF too large to fetch)* — higher per-image realism ≠ downstream utility; **coverage/diversity drives training value.** Citable support for our central finding.
- **CMMD (CVPR 2024)** [pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.pdf) — validates our metric choice; we'd be early using CMMD in the colony domain.

**Positioning / what to borrow:**
1. **Replicate-to-replicate whole-colony generation is underexplored** — prior work conditions on masks/sketches/simulations or composites single cells. Conditioning on one real colony to emit new replicates of *that* colony is closest to the non-biological "augmented conditioning" line — a defensible novelty. Nearest to cite/differentiate: biofilm npj-2025, colony-style-transfer 2022, and our own sim-templated preprint.
2. **Adopt the source-conditioning + explicit diversity-knob framing** (Augmented Conditioning) — it's the published version of our two-lever tradeoff.
3. **Borrow the field's utility currency:** report a **real-data-replacement / scarcity curve** ("X% real + synthetic = full baseline") on a segmentation/detection/feature-regression downstream, not just FID. And report **curated vs uncurated** synthetic (HistoGAN lesson).
4. **The "better backbone relaxes the tradeoff" hypothesis has real (not absolute) support** — CellFlux + DiADM show both axes can move together with a stronger/diversity-aware paradigm. Frame as "relax the frontier," not "eliminate it" (per DiADM/"Pretty isn't useful" caveats). Aligns with the Flux pilot.
5. **Two ownable contributions:** (a) an explicit realism↔diversity **Pareto curve tied to a downstream delta** for a colony augmenter (unreported in the bacterial papers); (b) **CMMD** as the distributional metric in this domain.

---

## Next steps this suggests
- **Downstream task (point 4):** a **colony-feature regression** head (targets 1–5 above) auto-labeled via Rattray-2023 code + FracLac, run as a scarcity curve — a cleaner "generative-should-win" testbed than the rotation-friendly multiplexed decode.
- **Backbone (points 1/5):** CellFlux is the citable case for flow-matching; strengthens the Flux pilot rationale.
- **Standing agentic watch:** point the same two-sweep harness at a monthly bioRxiv/arXiv query for colony + generative-augmentation to keep the competitor set current.
