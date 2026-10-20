# Target-domain data — outreach drafts

*Goal: get more real P. aeruginosa colony-replicate imagery, which is the "more data" the evidence says will actually improve the augmenter (broadening on off-morphology data didn't help). Two sources. Drafts for Sarthak to adapt/send — not sent automatically.*

---

## 1. Internal — EMBO / Luo branching-colony replicates (highest value)

**To:** Nan Luo (cc: Kinshuk)  ·  **Send via:** Kinshuk or directly, in Sarthak's voice.

> Hi Nan,
>
> I'm Sarthak, an undergrad working with Kinshuk in the You Lab on the generative data-augmentation project — a ControlNet + Stable Diffusion model that synthesizes realistic *P. aeruginosa* colony replicates to help with experimental data scarcity.
>
> We've shown the model adapts well when we fine-tune it on real replicate data of the target morphology, but the single biggest bottleneck is getting enough of that data. Your EMBO MSB 2021 paper ("Collective colony growth is optimized by branching pattern formation in *P. aeruginosa*") has exactly the dendritic/branching morphology we're targeting, with real biological replicates — but only the endpoint figures were published, and the GitHub repo (OptimalPatterns_NanLuo) is code-only.
>
> Would you be able to share the **raw, full-resolution colony images** behind that work — ideally the multiple biological replicates per condition (WT and the hyperswarmer/mutant lines), at whatever timepoints you captured? Even a subset would be very useful. We just need the images (any format); we handle preprocessing on our side.
>
> Happy to hop on a quick call, and of course we'll acknowledge/cite appropriately. Thanks so much!
>
> — Sarthak

**Why this one first:** exact morphology, real replicates, lab-owned (no license/access friction), and no open substitute exists (confirmed across three dataset searches).

---

## 2. External — full-resolution Rattray dataset (GaTech Brown Lab)

**To:** Jennifer B. Rattray / Prof. Sam P. Brown (GaTech, corresponding author)  ·  cite the paper.

> Dear Dr. Rattray and Prof. Brown,
>
> I'm an undergraduate researcher in the You Lab at Duke (BME), working on a generative model that synthesizes realistic *P. aeruginosa* colony images to augment scarce experimental datasets. Your dataset from "Machine learning identification of *Pseudomonas aeruginosa* strains from colony image data" (PLOS Comput. Biol. 2023) has been genuinely useful — the ~77 strains with quadruplicate biological replicates are a great fit, and we cite your work.
>
> The public copy in the GitHub repo is downsampled to ~256×256. For our diffusion model, higher resolution would meaningfully help. Would you be willing to share the **full-resolution original colony images** (the paper mentions ~12–183 MP captures)? A subset would be perfectly fine if the full set is large.
>
> We're using the data purely for research (training/evaluating the augmenter) and would gladly acknowledge the dataset and coordinate on any data-use terms you prefer.
>
> Thank you for making the work available — it's been a real help.
>
> Best regards,
> Sarthak  ·  You Lab, Duke University  ·  [email]

**Why:** same species, real replicates; full-res removes the 256px detail cap that limits the current Rattray fine-tune.

---

## After the data arrives
- Re-stage under `data/external_datasets/<source>/` with a README (same pattern as Rattray), build replicate pairs, and fine-tune the base — the pipeline is already in place (`reptorep_finetune_rattray.py`, held-out eval, tracking).
- Full-res data may also justify raising the pipeline resolution above 256px (a separate change to test).
