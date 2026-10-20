# Open-Source Datasets for the Replicate-to-Replicate Augmenter — Research Findings

---

## TL;DR

1. **Genuinely usable "paired replicate" data is scarce in the open.** Only 3 external sources offer paired/time-lapse data, and each has a real caveat.
2. **The single highest-value data is ours.** The You Lab's own *Luo et al. 2021* (EMBO MSB) P. aeruginosa **branching-colony** images are the exact target morphology with real biological replicates, but were never deposited as a dataset (figures only). **Best action: source that raw, full-res replicate data internally.**
3. **The two best *external* directly-usable sets are time-lapse swarming movies** (Proteus mirabilis; Enterobacter SM3) — on-modality, whole-plate — but their frames are 1 min apart, so they teach *incremental growth*, not *replicate-to-replicate diversity*.
4. **Unpaired pools (AGAR, MyceliumSeg) only help if we pretrain the base SD/VAE prior** — an unproven bet with a domain gap (RGB, fungal, round colonies). Worth a small experiment, not a big commitment.
5. **Several famous repos are dead ends here:** BBBC, DIBaS, and the 2026 clinical-bacteria set are all off-modality (single-cell microscopy or round ID colonies).

---

## Method note — the feasibility gate (why "a dataset exists" ≠ "it helps us")

*Confidence: medium — derived from ControlNet training practice + verifier caveats.*

Our ControlNet learns a **source → target** mapping between two replicates of the *same* experiment. That means:

- **Only PAIRED or TIME-LAPSE data is directly usable** as training pairs. An unpaired pile of colony photos cannot supply the replicate mapping.
- **Unpaired data can still help indirectly**, two ways:
  - **(a) Pretrain the base prior** — continue-train the SD 1.5 UNet/VAE on on-modality colony images so the backbone reconstructs bacterial-pattern textures faithfully *before* ControlNet conditioning.
  - **(b) Synthetic pairing** — turn one real image into a pseudo-replicate via elastic/rotation/growth-warp, or self-supervised degraded→clean pairs.
- **⚠️ Time-lapse caveat:** frames 1 minute apart are near-identical, so consecutive-frame pairs teach *how a colony grows*, not *how independent replicates differ*. Useful, but not equivalent to true biological replicates.

**Practical ordering:** Tier-1 (paired/time-lapse) for the ControlNet head → Tier-2/3 (unpaired) only for base-prior pretraining.

---

## Summary table (ranked by relevance)

| # | Dataset | Tier | Organism / modality | Paired? | Size / format | License | Verdict |
|---|---------|------|---------------------|---------|---------------|---------|---------|
| 1 | **You Lab — Luo 2021 (EMBO MSB)** | 1 (ideal morphology) | *P. aeruginosa* PA14 **branching/dendritic** colonies, whole-plate | Real replicates, but **endpoint-only, not deposited** | Figures only — no corpus | Paper CC-BY; **raw data internal** | ⭐ **Best fit — get it internally, not by download** |
| 2 | **Danino Lab — Proteus mirabilis engineered** | 1 | *P. mirabilis* bullseye **swarm** patterns, whole-plate scanner | **Time-lapse** (.mp4, ~12 frames, 1 fps) | 8 movies, **790×844 px** | ⚠️ No GitHub license; Zenodo "Other (Open)" | Usable; low-res, few frames |
| 3 | **SwarmEvo (HF)** | 1 | *Enterobacter* sp. SM3 **swarming**, whole-plate camera | **Time-lapse** (1-min interval) | 1250×1250 JPEG + masks | ⚠️ Conflicting (MIT vs "academic-only") | Usable; frames highly correlated |
| 4 | **GaTech Brown Lab — Rattray 2023** | 1 (structure) / off-morphology | *P. aeruginosa* **compact/rugose ID** colonies | **Genuine replicates** (mostly ×4) | 266+69 imgs; public = **224×224 RGB** | ⚠️ **Unconfirmed** | Real pairs but wrong morphology + downsampled |
| 5 | **AGAR** | 2 | 5 microbes (incl. *P. aeruginosa*, *B. subtilis*), whole-plate | No (single snapshots) | **18,000** imgs, color | CC **BY-NC** 2.0 | Base-prior pretraining only; compact colonies |
| 6 | **MyceliumSeg** | 3 | **Fungal** mycelium, whole-petri-dish | No (segmentation set) | **20,176** imgs, 4608×3456 RGB, ~53 GB | CC-**BY-4.0** ✅ | Large clean pool for texture prior; fungal, needs grayscale+resize |
| — | BBBC | ✗ | Eukaryotic single-cell/synthetic | — | — | — | **Skip** — off-modality |
| — | DIBaS | ✗ | Gram-stain 100× microscopy | — | 660 imgs | — | **Skip** — single-cell scale |
| — | Nature 2026 clinical bacteria | ✗ | 19 pathogens, **round ID** colonies | No (single timepoint) | 950 imgs | — | **Skip** — no replicates, wrong morphology |
| — | Paenibacillus vortex (Ingham 2008) | ✗ (ref only) | *P. vortex* fractal/vortex swarm | No | 10 figures | CC-BY | On-modality but **figures only**, not a dataset |

---

## Detailed entries

### TIER 1 — directly usable (paired / time-lapse)

**1. You Lab — Luo, Wang, Lu, Ouyang, You (2021), "Collective colony growth is optimized by branching pattern formation in *P. aeruginosa*."** *(our own lab's paper)*
- **Contents:** whole-plate photos of *P. aeruginosa* PA14 forming exactly our target **dendritic/branching + parallel-strip** patterns (UVP Colony Doc-It, 1 cm scale bar, 90–100 mm agar). Population-scale, not microscopy.
- **Replicates:** yes and biological (WT n=5, hyperswarmers n=10; also n=12/24, "replicated five times") — **but only representative *endpoint* images shown**, single terminal timepoint. No time-lapse, no released paired frame set.
- **Access:** GitHub `youlab/OptimalPatterns_NanLuo` holds **only MATLAB/Python code, zero raw images**. No Zenodo/Dryad/figshare accession.
- **Verdict:** the most on-point morphology anywhere, and **the lab already owns the raw data**. Action = source full-resolution replicate images *internally*, not scrape figures.
- Cite: https://www.embopress.org/doi/full/10.15252/msb.202010089 · https://pmc.ncbi.nlm.nih.gov/articles/PMC8073002/

**2. Danino Lab — "proteus-mirabilis-engineered" (Nat. Chem. Biol. 2023).**
- **Contents:** whole-plate, cm-scale *Proteus mirabilis* bullseye **swarm** patterns under gene knockouts (flatbed scanner, ~158 px/cm). Macroscopic, zero microscopy.
- **Paired/time-lapse:** yes — `Low_res_timelapse_movies/` = 8 `.mp4` scanner time-lapses (verified 790×844 px, ~12 frames, 1 fps); consecutive frames → source→target pairs. Full high-res (100s of GB) on request.
- **License:** ⚠️ GitHub has **no LICENSE file**; Zenodo lists "Other (Open)" — verify before publishing derivatives.
- **Verdict:** Tier 1, on-modality swarming. Caveats: low resolution (bottom of our ~hundreds–2000 px target) and only ~12 frames/movie.
- Cite: https://github.com/daninolab/proteus-mirabilis-engineered · DOI 10.5281/zenodo.7637609 · https://www.nature.com/articles/s41589-023-01325-2

**3. SwarmEvo (HuggingFace `ShengyouDuan/SwarmEvo`).**
- **Contents:** *Enterobacter* sp. SM3 **swarming** colony expansion, whole-plate (vertical camera + uniform LED). Not microscopy.
- **Paired/time-lapse:** yes — `raw_imgs/` ships per-colony consecutive JPEG frames at fixed **1-minute** intervals (native 1250×1250 px); also YOLO polygons + binary masks (its primary task is mask forecasting).
- **License:** ⚠️ **contradictory** — HF metadata says MIT, README + paper say "academic research use only." Treat as academic/non-commercial.
- **Verdict:** Tier 1, on-modality. Caveats: 1-min frames highly correlated (growth, not replicate diversity); mask-centric framing.
- Cite: https://huggingface.co/datasets/ShengyouDuan/SwarmEvo · https://arxiv.org/abs/2602.01056

**4. GaTech Brown Lab — Rattray et al. 2023 (PLOS Comput. Biol.).**
- **Contents:** whole-colony (cm-scale, raw 12M–183M px) RGB photos of *P. aeruginosa* on Congo-Red LB agar. Population-scale (4× + stitching is just capture).
- **Replicates:** ✅ genuine — each strain imaged mostly in **quadruplicate** (some ×3/×2) → real same-condition pairs (266 train/val + 69 test).
- **⚠️ License REFUTED:** the "openly available under permissive CC, directly usable" claim was **refuted 0-3** — the data repo has **no LICENSE file** (parent article is CC-BY, but the data files aren't clearly licensed). Public copy is **downsampled to 224×224** and RGB.
- **Verdict:** Tier 1 by *structure*, but morphology = **compact/wrinkled strain-ID colonies** (the kind we explicitly excluded), not dendritic. Use cautiously; confirm license with authors.
- Cite: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011699 · https://github.com/GaTechBrownLab/Rattray-2023-PLOSCompBio

### TIER 2 — unpaired, on-modality (base-prior pretraining)

**5. AGAR — "Annotated Germs for Automated Recognition."**
- **18,000** whole-agar-plate photos of 5 microbes (incl. *P. aeruginosa*, *B. subtilis*), single/mixed cultures, 2 cameras. Object-detection labels (336,442 colonies), **no replicates, no time-lapse**.
- **License:** CC **BY-NC** 2.0 (non-commercial). Compact countable colonies → value is texture pretraining only.
- Cite: https://agar.neurosys.com/ · https://arxiv.org/abs/2108.01234

### TIER 3 — broader pattern priors

**6. MyceliumSeg (Zenodo 15224240; Nat. Sci. Data 2025).**
- **20,176** macroscopic whole-petri-dish **fungal** mycelium images (567 mask-labeled + **19,609 unlabeled**), 4608×3456 RGB JPG, ~53 GB. Germination → full-dish coverage.
- **License:** CC-**BY-4.0** ✅ (cleanest license here). Large permissive unlabeled pool for organic branching/space-filling growth textures.
- **Caveat:** fungal (not bacterial), RGB + very high-res → convert to grayscale + resize.
- Cite: https://zenodo.org/records/15224240 · https://www.nature.com/articles/s41597-025-06265-1

### Negative results — verified dead ends
- **BBBC** (Broad Bioimage Benchmark): all eukaryotic single-cell/synthetic; zero bacterial-colony/pattern data. https://bbbc.broadinstitute.org/
- **DIBaS:** Gram-stained bright-field **100× microscopy** (single cells: cocci vs bacilli), 660 imgs. Off-scale.
- **Nature 2026 "High-Resolution Colony Images of Clinically Isolated Bacteria":** 950 imgs of 19 pathogens as **round ID colonies**, single timepoint, no paired replicates (mirror/flip aug ≠ biological replicate). https://www.nature.com/articles/s41597-026-07095-5
- **Paenibacillus vortex (Ingham & Ben-Jacob 2008):** on-modality vortex/fractal swarm morphology, but **figures only** — no supplementary image corpus or deposited movies. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2268691/

---

## Caveats & risks
- **Licensing is the biggest open risk.** 3 of 4 Tier-1 sources have unclear/absent licenses (GaTech: none; Danino: none on GitHub / "Other (Open)"; SwarmEvo: MIT-vs-academic conflict). AGAR = BY-NC (non-commercial). MyceliumSeg = clean CC-BY-4.0. **Confirm before using anything in a publishable/released model.**
- **Tier-1 scarcity is real:** the two time-lapse sets teach growth not diversity; the one true-replicate set is off-morphology + 224×224.
- **Modality mismatches:** most pools are RGB and often very high-res (need grayscale + resize); MyceliumSeg is fungal.
- **Coverage gap (honest):** several listed repos — **IEEE DataPort, EMBL-EBI BioImage Archive, IDR, Cell Image Library, OSF, Dryad, Kaggle, Figshare** — produced **no verified hits this round.** That reflects *lack of verification*, not proven absence; they're worth a second targeted pass.

## Open questions for discussion
1. Can we pull **full-resolution** versions of the two most relevant paired sources — the **You Lab's own EMBO branching-colony replicates (internal)** and the **GaTech raw 12M–183M px** images — since public copies are figure-only or 224×224 (below our ~hundreds–2000 px target)?
2. What are the **author-confirmed licenses** for the 3 unclear Tier-1 candidates, and are they OK for a model we may later publish/release?
3. **Empirical bet:** does pretraining the SD 1.5 prior on unpaired pools (AGAR + MyceliumSeg) *measurably* improve OOD replicate synthesis, or does the domain gap (fungal, round colonies, RGB) just add noise? Worth GPU time?
4. Do the **untapped repos** (BioImage Archive, IDR, IEEE DataPort, Dryad, + more *B. subtilis*/*Paenibacillus dendritiformis*/*Proteus* pattern papers) hide any Tier-1 branching/swarming sets this round missed?

---

# ROUND 2 — Second liberal pass (2026-09-03)

*Per Lingchong's guidance (be liberal on candidates, then check for replicate data). Deep multi-source search — 5 angles, 17 sources fetched, 25 claims adversarially verified (3-vote) — across the repos Round 1 never checked: BioImage Archive, IDR, Dryad, OSF, Zenodo, Kaggle, figshare, Mendeley. Two candidates were **downloaded and visually inspected** (samples in `model_results/dataset_samples_round2/`).*

**Bottom line: still no open dataset of the target morphology (dendritic/branching *P. aeruginosa* whole-plate). The highest-value lever remains the internal You Lab EMBO data (Luo 2021).** The new finds are clean-license but partial / off-morphology.

## New, directly-downloadable, verified

| Dataset | Organism / morphology | Replicates / time-lapse | Size / format | License | Link | Verdict |
|---|---|---|---|---|---|---|
| **Myxococcus xanthus development** (BioImage Archive S-BIAD2328) | fruiting-body **aggregation**, mm-scale dissecting-scope — NOT dendritic | ⭐ time-lapse 1441 frames/movie (1-min); dedicated **WT replicate set** (multiple "scopes") + many mutants | WT set = 105 MB zip (1296×972 JPG); full study larger | **CC0** ✅ | ebi.ac.uk/biostudies/studies/S-BIAD2328 | **Inspected → low morphology match** (field of small aggregates, not colony patterns) |
| **B. subtilis biofilm** (Dryad dryad.f4qrfj71n) | *B. subtilis* **wrinkled biofilm** (target genus, wrong morphology) | replicates present, unquantified | only ~677 MB of 49.7 GB is macroscopic (rest confocal) | **CC0** ✅ | datadryad.org/dataset/doi:10.5061/dryad.f4qrfj71n | Partial; must pull 50 GB for 677 MB useful; not dendritic |
| **Yeast colony time-lapse** (figshare 15040353) | *S. cerevisiae* (**fungus**), grid-arrayed whole colonies | ⭐ dense time-lapse (~187 frames/colony) | 3.4 GB zip, JPEG | **CC BY 4.0** ✅ | doi.org/10.6084/m9.figshare.15040353 | Clean + rich but fungus → low relevance |
| **B. mycoides/subtilis colonies** (figshare 1133794) | small **spot/grid** colonies (faint rhizoid edges) | day1/2/3 × ~5–6 each | 73 MB, 2048×2048 16-bit TIF | **CC BY 4.0** ✅ | doi.org/10.6084/m9.figshare.1133794 | **Inspected → low relevance** (tiny arrayed colonies, not dendritic) |

## Re-verified (already known) — with corrections
- **SwarmEvo** (HF `ShengyouDuan/SwarmEvo`): still the **best external branching match** — *Enterobacter* swarming, finger-like fronts, macroscopic whole-plate, 1,971 seg + 276 time-series across 3×3×3 conditions, ~15 GB. ⚠️ **License conflict confirmed** (MIT tag vs "academic-use-only" README) — treat as restricted-academic; resolve with authors before any released model.

## Confirmed dead ends (drop these)
- **IDR** — verified empty of colony imagery (only cellular microscopy). Drop from future searches.
- **Physarum Dryad** (dryad.0k6djhb9m) — CC0 but no network images deposited (only viscometer blobs + CSVs).
- **Xiong 2020 eLife "flower patterns"** (e48885) — relevant morphology + time-lapse *videos*, but figures/videos only, no deposited dataset → author request only.
- **arXiv 2501.02855 "Synthetic Fungi"** — synthetic vector drawings, not real images.

## Still uncovered (likely figures-only; a 3rd pass would mean author requests)
*Paenibacillus dendritiformis*/vortex, *Proteus* swarm (beyond the known Danino set), *Serratia*, swarming *E. coli*/*Salmonella*.

## Recommendation
1. **#1 lever — get the internal EMBO branching-colony replicates** (Luo 2021, via Nan Luo / Kinshuk). Exact morphology, real replicates, lab-owned. No open substitute exists (confirmed across two thorough passes).
2. **Since fine-tuning works**, **SwarmEvo** (branching swarm) is the best external set to broaden the prior / test cross-morphology generalization — pending license resolution.
3. The CC0 sets (Myxococcus, B. subtilis) are clean-license options for **base-prior pretraining only**; partial morphology, unproven payoff.
4. **Blocker:** DCC `youlab` group disk is full — free space before staging any external set.

---

# ROUND 3 — Widened criteria (2026-09-03): *any* microbial spatial growth, license not a filter

*Updated scope (Kinshuk/Lingchong): morphology no longer restricted to branching — ANY microbial spatial growth (colonies / biofilms / swarms / mats / aggregation / mycelium) qualifies; and **any published, citable dataset is usable regardless of license** (cite on use). This both re-scopes earlier finds as usable AND opens high-volume sources. Deep verified pass — 5 angles, 20 sources, 25 claims 3-vote (18 confirmed, 7 overclaims killed).*

## New, directly downloadable (verified)
| Dataset | Organism / growth | Structure | Size / res | License | Link |
|---|---|---|---|---|---|
| **Clinical urine-culture plates** | mixed clinical bacteria, whole-plate colonies | 1,500 single snapshots (498 pos / 500 neg / 502 uncertain) | 3024×3024 JPEG | CC BY 4.0 | Mendeley `xrbtd74pfj/3` |
| **NTM Mycobacterium colonies** | 10 NTM species, whole-plate | 92 originals (+92 aug), multi-isolate | iPhone JPG | CC BY 4.0 | Mendeley `6fy7gdsr2k/2` |
| **Yeast colony time-lapse** (link now verified) | *S. cerevisiae* colonies | 5,500 colonies / 196 strains, ~1 img/23 min ×3 d | **100×100 px** JPEG, 3.63 GB | CC BY 4.0 | figshare 15040353 |

## New, on-modality — extract-from-video
| Source | Organism / pattern | Cadence | Access |
|---|---|---|---|
| **B. subtilis biofilm time-lapse** (Sci Rep 2022) | 5 strains, wrinkled biofilm, mesoscopic | 6-min, 3–21 d | 10 figshare/YouTube videos |
| **Capnocytophaga ochracea swarming** (ISME J 2025) | **flower/chrysanthemum swarm patterns** | 5-min, 3–4 d | 13 AVI movies |
| **Xiong 2019 flower patterns** (eLife 48885) | *Acinetobacter* + *E. coli* flower colonies | time-lapse | 8 eLife videos |

## Killed / excluded (adversarial check did its job)
- figshare 22022540 "bacterial colony detection" — Zenodo download + 950-image claims **refuted**.
- Cell Reports 2016 *B. subtilis* branching — not usable imagery (**refuted**).
- Dryad `pg4f4qrsw` — white-light-interferometry **height CSVs, not photos**.
- Mendeley `psk2k6td6p` — spreadsheet only, **zero images**.
- NTM "weekly 7-week time-course" — **refuted** (not a time series).

---

# ⭐ MASTER SHORTLIST (all 3 rounds, under relaxed criteria) — what to actually use

Organized by how each feeds the **replicate-to-replicate ControlNet**:

### A. TRUE replicate pairs — direct source→target (best)
- **Rattray-2023** — *P. aeruginosa* (**target species**), ~77 strains × 4 real biological replicates, **256 px, background already removed**. ✅ verified + sampled, 15 MB, directly usable. **Start here.**

### B. Time-lapse — consecutive-frame growth pairs
- **SwarmEvo** (*Enterobacter* swarm, whole-plate, ~15 GB) · **Danino Proteus** (swarm .mp4) · **Myxococcus** (CC0 aggregation) · **yeast** (100 px) · video sets (B. subtilis biofilm, Capnocytophaga, Xiong).

### C. Volume / diversity — base-prior pretrain OR synthetic pairing (via the existing rotation-AUG pipeline)
- **AGAR** (18 k, 5 microbes incl. Pa/Bs) · **urine plates** (1,500 @ 3024 px) · **MyceliumSeg** (20 k fungal) · **Nature clinical** (950, 19 species) · **NTM** (92, 10 species) · **B. subtilis biofilm Dryad** (677 MB macroscopic subset).

## Bottom line
Under the relaxed criteria there is now **plenty of usable data**, with one method caveat: only **Rattray gives true same-species replicate pairs at our native resolution** — the rest are time-lapse (growth pairs) or single snapshots (base-prior / need synthetic pairing). The **internal EMBO branching data remains the single best morphology + replicate match** if obtainable. Practical plan: **(1)** broaden/fine-tune on **Rattray** now (tiny, verified, real replicates); **(2)** add **SwarmEvo + AGAR** for breadth once DCC disk is freed; **(3)** treat video sets as optional frame-extraction later.
