# DojoGrants (Nucleate Dojo) — Application Package

**Applicant:** Sarthak · **Lab:** You Lab, Duke University · **Prepared:** 2026-08-11
**Form:** DojoGrants Research Fellowship Application (2026–2027) — $5,000 undergraduate research grant

> Purpose of this doc: (1) a super-detailed, accurate description of the You Lab research you can draw
> from to answer any research question, and (2) every question on the form, with drafts for the
> research/funding items and clear `[NEEDS YOU / RESUME AGENT]` tags for the personal ones.

---

## 0. READ FIRST — eligibility & logistics flags

- **Undergraduate-only.** ✓ You confirmed you're an undergraduate — eligible on this criterion.
- **Citizenship/visa.** ✓ You're a U.S. citizen — eligible. On the form, select "U.S. citizen or
  permanent resident."
- **Final deadline: August 14, 2026, 11:59 PM PT.** Today is Aug 11 → ~3 days. (Priority deadline
  Aug 7 has passed, so you'll be reviewed "if reviewer capacity allows." Worth applying anyway.)
- **Supervisor letter.** After you submit, your listed supervisor gets an auto-email with a unique
  upload link. Their letter deadline is **September 1, 2026**. → **Message your supervisor NOW** so
  they expect it and the email doesn't hit spam.
- **Authenticity rule (important).** The form explicitly says: *do not paste AI-generated answers.*
  Reviewers screen for generic/AI text. **Every draft below is scaffolding — rewrite the personal
  statements in your own voice**, with details only you know. Use these to get unstuck, not to copy.
- **$5k restrictions.** Cannot fund tuition, loans, phone bills, apartment utilities, or anything not
  tied to research. Rent/food only if required for the research and not already covered by a
  university plan. Must be a *credible, itemized budget*.
- **Winners announced:** September 11, 2026. Project must run during the 2026–2027 academic year,
  ≥6 months at 10–20 hrs/wk (part-time) or ≥4 months at 40 hrs/wk (full-time).

---

## 1. What DojoGrants is (context for your resume agent)

Nucleate Dojo is the undergraduate-focused arm of **Nucleate**, a global nonprofit for
bioentrepreneurs. DojoGrants gives **$5,000** to undergraduates to pursue **life-sciences / biotech
research** in the U.S. It targets students *whose access to research would be meaningfully
strengthened by funding* (e.g., unpaid/underfunded research, students balancing work or family,
first-time researchers, ambitious projects with limited institutional support).

Reviewers look for: **research promise, initiative, persistence/independence, ownership of hard
problems, follow-through, and a clear explanation of how the funding changes what you can do.** You
do *not* need prior publications or awards. Program perks (useful for the "how it helps" answer):
mentorship from Nucleate members/PhDs/industry, professional-development workshops, founder/industry
speaker sessions, the Nucleate network (researchers, founders, investors), cohort community, and a
virtual **Annual Symposium** with formal talks + poster sessions.

---

## 2. SUPER-DETAILED RESEARCH DESCRIPTION (You Lab)

### 2.1 One-paragraph plain-English summary
The You Lab (PI **Dr. Lingchong You**, Duke Biomedical Engineering) studies how engineered and
natural bacteria **self-organize into spatial patterns** — e.g., the branching, dendritic colonies
formed by *Pseudomonas aeruginosa*. Studying these patterns experimentally is slow and hard to
parallelize, so there's very little training data for machine learning. This research builds a
pipeline that **learns from physics-based simulations to predict and generate realistic bacterial
patterns**, combining reaction–diffusion simulations with modern generative AI (diffusion models /
ControlNet). The payoff: predict experimental outcomes in seconds instead of weeks, and use
**generative data augmentation** to overcome the biological data-scarcity bottleneck. The specific
project this application supports is the **generative data-augmentation** thread — teaching a model
to synthesize realistic experimental "replicates" so downstream pattern-prediction models can train
on far more data than the wet lab can produce.

### 2.2 The scientific problem
- **Self-organized pattern formation** is fundamental in biology (development, biofilms, microbial
  communities). *P. aeruginosa* colonies form diverse structures — circular colonies, dense
  dendritic branches — over centimeter scales, depending on nutrients, signaling, and seeding.
- **Data scarcity is the bottleneck.** Each experimental pattern takes significant bench time and
  can't be mass-produced, so there aren't enough images to train data-hungry ML models. This blocks
  predictive, quantitative modeling of these systems.
- **The reality gap.** Physics simulations are fast and controllable but look "too clean";
  experiments are noisy and stochastic. Bridging simulation → experiment is an open problem.

### 2.3 The approach / full pipeline
Two connected projects (folders in the working tree):
`Simulation_templated_pattern_prediction-main` (the core method) and `Data-augmentation-main`
(the extension this grant supports).

**(a) Physics simulations (MATLAB).** A reaction–diffusion + branching-growth model (extended from
Luo et al., *Mol. Syst. Biol.* 2021) generates ~10⁵ synthetic patterns across parameter regimes and
random seedings. This is the cheap, controllable "ground truth" the ML learns from.

**(b) Deterministic accelerator — dilated ResNet.** A PDEArena-style **dilated ResNet** (15 residual
blocks, dilation rates 1/2/4/8 for multi-scale receptive fields) maps an initial *seed* image to
later-time / different-parameter patterns in one forward pass — **orders of magnitude faster than
running the MATLAB simulation.** It operates in the compressed latent space of a Stable Diffusion VAE
(32×32×4 latents) for efficiency. Trained on tens of thousands of simulated patterns; evaluated with
**SSIM** (structural similarity).

**(c) Generative sim→exp translation — ControlNet + Stable Diffusion.** To cross the reality gap, a
**ControlNet** adapter conditions a frozen **Stable Diffusion v1.5** on a simulation/seed image and
generates a *stochastic, experiment-like* pattern. Only the ControlNet is trained (backbone locked),
using ~40k augmented simulation↔experiment pairs; evaluated with **SSIM, LPIPS** (perceptual
similarity), ORB feature matching, and a Siamese-network dissimilarity score.

**(d) Inverse problem — decoding.** A U-Net (+ optional frozen VAE) recovers the **initial seeding
configuration** from an observed pattern (sim→seed and exp→seed), linking phenotype back to the
conditions that produced it.

**(e) Generative data augmentation (`Data-augmentation-main`) — the focus of this grant.** Instead of
hand-crafted rotations/crops, a **replicate-to-replicate ("rep-to-rep") ControlNet model** learns to
generate *new, realistic experimental replicates* from an existing one, capturing the natural
biological variation between replicates. Trained across multiple *P. aeruginosa* datasets (Sahu et
al., Şimşek et al., Lu et al.); rotational augmentation expands the set to ~**300,000 images**. This
is *model-agnostic* augmentation: it produces more training data for any downstream pattern-prediction
model, directly attacking the data-scarcity bottleneck.

### 2.4 Where this stands / preliminary results
- Deterministic dilated-ResNet predicts seed→pattern with high fidelity (SSIM strong vs. ground
  truth) at a fraction of simulation cost.
- ControlNet sim→exp generation qualitatively reproduces experimental morphology and its
  stochastic variation.
- Rep-to-rep augmentation reproduces replicate variation well **in-distribution**, but **struggles on
  out-of-distribution species** not seen in training — a clear, concrete open problem.
- The broader method is written up in a manuscript from the lab (**Sahu, Davis, Lu, Villalobos,
  Heyman, Şimşek, You**, "Mechanistically conditioned generative modeling of biologically realistic
  bacterial patterns," bioRxiv). *Note: this is the existing lab work you're building on — you are a
  new contributor extending the data-augmentation thread, not an author of that manuscript. Represent
  it that way.*

### 2.5 Your role / the project this grant funds  `(confirmed with you)`
Your mentor, PhD student **Kinshuk Sahu**, has assigned you the **generative data-augmentation
project** for now — the plan is to build from it into larger lab projects across the year, and your
own goal is to learn as much computational biology as you can along the way. A concrete, fundable
scope for the data-augmentation work:
- Get the rep-to-rep augmentation pipeline running in your own compute environment (in progress: you
  were granted push access to the `Data-augmentation` repo and are standing it up on Duke's cluster).
- **Improve out-of-distribution generalization** — make the augmenter produce realistic replicates
  for bacterial species/conditions it wasn't trained on (the current failure mode).
- **Add quantitative evaluation** (SSIM/LPIPS + the Siamese dissimilarity metric) to replace
  qualitative "looks right" judgments.
- **Close the loop:** show that training a downstream pattern-prediction model on the synthetic
  augmented data measurably improves accuracy vs. real data alone — the claim that makes generative
  augmentation useful — then use that foundation to move into the larger projects planned for later
  in the year.

> `[OPTIONAL]` If you can name one specific target this year (a species, a benchmark, a figure), drop
> it in — that specificity is what reviewers reward.

### 2.6 Why it matters (grounded — avoid hype)
- **Data efficiency for biology ML.** Generative augmentation lets expensive wet-lab experiments go
  further, a general problem across the life sciences.
- **Fast, predictive modeling.** Learned surrogates replace slow simulations, enabling rapid
  screening of conditions in silico before committing bench time.
- **Translational directions (state as possibilities, not promises):** optimizing biofilm/colony
  morphology in **biomanufacturing/fermentation**; **colony-morphology-based diagnostics** (pattern
  as a readable signature of state); and **inverse design** of engineered bacteria with target
  spatial behaviors in synthetic biology. Open-sourcing models (Hugging Face) would let other labs
  reuse the method.

### 2.7 Tech stack & compute
Python 3.10, PyTorch 2.0.1 (CUDA 11.8), Hugging Face Diffusers 0.20 + Transformers, PyTorch
Lightning, a modified `lllyasviel/ControlNet`, OpenCV/scikit-image, LPIPS / pytorch-msssim / FID for
metrics, Optuna + TensorBoard, Gradio for demo UIs; MATLAB for the physics simulator. Runs on the
**Duke Compute Cluster (DCC)** via SLURM — the You Lab's `youlab-gpu` partition (nodes with 4×
NVIDIA A5000 GPUs, 48 CPUs, ~515 GB RAM). Datasets/checkpoints live in lab group storage; base
Stable Diffusion checkpoint (~4 GB) plus large augmented image sets (hundreds of thousands of images)
drive real storage/compute needs.

### 2.8 Mini-glossary (for reviewers who aren't ML people)
- **Reaction–diffusion model:** equations describing how chemicals spread and react to produce
  spatial patterns — the physics behind bacterial branching.
- **Diffusion model / Stable Diffusion:** a generative AI that creates images by iteratively
  denoising; here used to synthesize realistic bacterial patterns.
- **ControlNet:** an add-on that steers a diffusion model with a conditioning image (a simulation or
  a seed layout) so the output matches a desired structure.
- **VAE latent space:** a compressed numerical representation of an image; models work there for
  speed.
- **SSIM / LPIPS:** metrics for how similar two images are (structural / perceptual).
- **Out-of-distribution:** inputs unlike anything in training — where models tend to fail.

---

## 3. THE APPLICATION — every question, with drafts & tags

Legend: `[DRAFT]` = written below, personalize it · `[RESUME AGENT]` = hand to your resume agent ·
`[YOU]` = only you can answer · `[CONFIRM]` = verify a fact.

### Section A — Applicant Information & Demographics
1. **Full legal name** — `[YOU]`
2. **Preferred name (optional)** — `[YOU]`
3. **Preferred pronouns** — `[YOU]` (She/Her · He/Him · They/Them · Other · Prefer not to say)
4. **Race/ethnicity (multi-select)** — `[YOU]`
5. **Do you identify with any of: disability / LGBTQ+ / refugee-IDP / rural / none / prefer not** — `[YOU]`
6. **Citizenship/residency/visa status** — `[YOU / CONFIRM eligibility]`
7. **Institution email address** — `[YOU]` use your `@duke.edu` email (add dojo@nucleate.org to contacts)
8. **University + city/country** — `[DRAFT]` → `Duke University (Durham, North Carolina)`
9. **Institution type (multi-select)** — `State or public university` → actually Duke is **private**;
   likely `None of the above`. `[CONFIRM]`
10. **First-gen / homeless / foster / none (multi-select)** — `[YOU]`
11. **Household income bracket** — `[YOU]`
12. **Field of study (majors/minors/specializations)** — `[RESUME AGENT / YOU]`
13. **Undergrad year (2025–2026)** — `[YOU]`
14. **Anticipated graduation (Month Year)** — `[RESUME AGENT / YOU]`

### Section B — Research Opportunity Information
15. **Name/location/hyperlink of the U.S. research institution/lab** — `[DRAFT]`
    > You Lab (PI: Dr. Lingchong You), Department of Biomedical Engineering, Duke University
    > (Durham, NC). Affiliated with the Duke Center for Genomic and Computational Biology.
    > Link: https://bme.duke.edu/people/lingchong-you/
    > `[CONFIRM if the lab has a dedicated site you prefer to link]`

16. **Scope of the research opportunity (100 words max)** — `[DRAFT — personalize]`
    > In Dr. Lingchong You's lab at Duke, I work on machine-learning models of bacterial
    > self-organization — how *Pseudomonas aeruginosa* forms branching colony patterns. My focus is
    > generative **data augmentation**: training a ControlNet/diffusion model to synthesize realistic
    > experimental "replicates" so that pattern-prediction models can train on far more data than the
    > wet lab can produce. Concretely, I'm improving the model's generalization to bacterial species
    > it wasn't trained on, adding quantitative image-similarity evaluation (SSIM/LPIPS), and testing
    > whether synthetic data measurably improves a downstream predictor. The work runs on Duke's GPU
    > cluster in PyTorch. `(~95 words — trim/adjust to your real assignment)`

17. **Approx. start–end date (Month Year – Month Year)** — `[YOU]`
    > Suggested: `September 2026 – March 2027` (≥6 months, part-time) or match your real timeline.
    > Must fall in the 2026–2027 academic year.
18. **Hours per week** — `[YOU]` (part-time band is 10–20 hrs/wk; full-time is 40)
19. **Research supervisor full name** — `Kinshuk Sahu` (PhD student, You Lab — your direct mentor)
    > A PhD mentor is a valid "research supervisor" here; the form only needs someone who can confirm
    > your role, involvement, and hours. Kinshuk assigned and supervises your project, so his letter
    > will be specific and credible. (Optional: note that Dr. You is aware — not required.)
20. **Research supervisor email** — `[YOU — send me Kinshuk's email]` Don't guess it on the form. Give
    Kinshuk a heads-up now that he'll get an automated Nucleate upload link after you submit
    (his letter is due Sep 1, so flag it early so it doesn't sit in spam).

### Section C — Personal Statements  *(write these in YOUR voice — anti-AI screening)*

21. **One project/initiative/problem you took ownership of — what you did, what was hard, what you
    learned. (1500 char)** — `[DRAFT — a strong true option; swap if you have a better story]`
    > When I joined the You Lab, the generative-augmentation code I needed to build on was written by
    > a PhD student and hard-wired to *his* cluster paths, checkpoints, and environment — it wouldn't
    > run for anyone else. I took ownership of making it reproducible for me. I set up passwordless,
    > multiplexed SSH to Duke's compute cluster, rebuilt the conda/PyTorch+CUDA environment from
    > scratch, and traced why training crashed — configs silently defaulted to inaccessible paths, so
    > I re-pointed them with environment variables and located the model checkpoints and datasets in
    > shared storage. The hardest part wasn't any single bug; it was that nothing was documented, so
    > I had to read unfamiliar code and cluster/SLURM behavior carefully enough to know *which* layer
    > was failing. I learned to debug systems I didn't write, to value reproducibility, and that
    > getting a pipeline to actually run is real research work, not just setup. `(~1,050 chars — edit
    > to your truth; add a concrete number or "aha" moment if you can)`

22. **What research question/problem/field most interests you right now, and why? (1500 char)** — `[DRAFT]`
    > I'm most interested in a bottleneck that limits machine learning across biology: there's rarely
    > enough experimental data. In the You Lab I see it directly — each bacterial pattern takes real
    > bench time, so we can't produce the tens of thousands of images modern models want. The question
    > I keep circling is: *can a generative model learn the natural variation in biological data well
    > enough to create new, realistic examples that actually make downstream models better?* What hooks
    > me is that it sits between physics and deep learning — reaction–diffusion simulations give
    > controllable "ground truth," and diffusion models add the messiness of real experiments. It also
    > has a clean failure mode I can attack: the augmenter breaks on species it hasn't seen, which
    > tells me it's memorizing rather than understanding variation. I care about this because if it
    > works, it generalizes far beyond bacteria — any field where experiments are slow and data is
    > scarce. `(~1,020 chars — make the "why" personal: what first made you care?)`

23. **Next major research/career goals + how DojoGrants helps. (2000 char)** — `[DRAFT — personalize]`
    > Right now my biggest goal is to learn as much as I can — I'm genuinely excited about
    > computational biology, and this project is my way into it. Concretely, I want to use this year to
    > go from *setting up* the data-augmentation project to *taking ownership* of it: getting the
    > generative augmenter to work on bacterial species it currently fails on, adding real quantitative
    > evaluation, and building the foundation for the larger projects my mentor and I plan to grow into
    > as the year goes on. Along the way I want a first concrete research output — a poster or write-up
    > I can present — and a clearer sense of whether I pursue computational-biology research long term
    > (I expect I will). DojoGrants would help on both fronts. The **funding** removes the practical
    > limits on an unpaid project: compute to run bigger experiments, travel to present my work for the
    > first time, and a research course to build skills faster. The **program** is almost tailor-made
    > for where I am: mentorship from PhD students and industry members to guide what I learn next;
    > presentation practice and the Annual Symposium to give my first research talk; founder and
    > industry sessions plus the Nucleate network to see how computational biology turns into real
    > tools and ventures; and a peer cohort to learn alongside. `(~1,380/2000 chars — add one specific
    > milestone or a named course/conference if you can; keep it in your own voice)`

24. **Optional: anything else about your background/circumstances/path (2500 char)** — `[YOU, optional]`
    > Good place for: unusual academic path, why you came to research late/independently, institutional
    > limitations, another initiative you're proud of, or context on financial need. Leave blank if
    > nothing applies.

### Section D — Materials & Links
25. **CV/Resume (PDF, no page limit) — required** — `[RESUME AGENT]`
26. **LinkedIn URL (optional)** — `[RESUME AGENT / YOU]`
27. **Portfolio/personal site/"anything cool you've built" (optional)** — `[YOU]` (GitHub is great here —
    e.g., your `Data-augmentation` fork, if you're comfortable sharing)

### Section E — Funding Request  *(see full budget in §4)*
28. **How would this funding support your ability to participate in/advance your research? (tie to
    access/participation/output)** — `[DRAFT, see §4 narrative]`
29. **Itemized $5,000 budget (category, amount, timing, how it supports the research; note paid/unpaid
    and course credit)** — `[DRAFT, see §4]`
30. **Other funding sources (wages, stipends, scholarships, lab funding, housing, family, credit…)** —
    `[YOU]` — be honest; if none, leave blank.
31. **Currently receiving need-based aid? (need-based financial aid / gov aid / scholarships / none)** —
    `[YOU]`
32. **Optional: anything your financial situation questions didn't capture** — `[YOU, optional]`

### Section F — Acknowledgements (check all) + Source
33–37. **Five certification checkboxes** — read & check each `[YOU]`.
38. **How did you hear about the program?** — `[YOU]` (you found it via **LinkedIn**, per your link).

---

## 4. PROPOSED $5,000 BUDGET  *(finalized: unpaid research → research stipend + costs)*

You earn no income and will commit ~20 hrs/week to unpaid research, so the largest line is a
**research stipend** for that time. Framing point that matters: DojoGrants' own example funds a
student's time via an hourly wage, so this is allowed — but their example says "reduce paid work
hours," which assumes you *have* a paid job to cut. You don't, so **frame it as a stipend for research
participation**, not as reducing hours you don't work. Compensating your time so you can sustain
unpaid research is explicitly a valid use ("access or continue unpaid research").

### Funding narrative (Q28) — `[DRAFT — personalize]`
> My research in the You Lab is **unpaid and not for course credit**, and I currently earn no income.
> I plan to commit about **20 hours a week** to the project during the 2026–2027 year. The largest
> part of this grant would act as a **research stipend** (calculated at $15/hour) that compensates my
> time, so I can sustain a serious weekly commitment to research instead of having to prioritize paid
> work. My compute is provided free by the university (the Duke Compute Cluster), so the remainder
> covers only the direct costs of *sharing and advancing* the work: travel to present my results for
> the first time, plus modest training and materials. Every item ties to research participation,
> progress, or output — not general living costs.

| Item | Amount | Timing | How it supports the research |
|---|---|---|---|
| Research stipend (unpaid research) | **$4,000** | Sep 2026–Mar 2027 | Research is unpaid & not for course credit and I earn no income. Partial stipend at $15/hr for my ~20 hrs/week commitment (~265 hrs), so I can sustain the time on research rather than paid work. |
| Conference registration + travel | **$800** | Spring 2027 | Register + travel to present a poster at **ISMB (Intelligent Systems for Molecular Biology)**, at its Undergraduate Research Symposium — my first research presentation. |
| Comp-bio / ML training | **$100** | Winter 2027 | Coursera **DeepLearning.AI "GANs Specialization"** (~$49/mo, ~2 months) — builds the generative-image-modeling foundations my ControlNet/diffusion work relies on. |
| Poster, figures & research materials | **$100** | 2026–2027 | Poster printing/figures and any small software or dataset costs to produce and share the work. |
| **Total** | **$5,000** | | |

**Budget notes**
- **Stipend framing matters.** Say "research stipend for participating in unpaid research," NOT
  "reduce paid work hours" (you have none). Both compensate research time, but only the stipend
  wording is accurate for you.
- **No compute line** — the Duke Compute Cluster is free to you, so billing the grant for it would be
  dishonest. Good that you flagged it; that's why the stipend is now $4,000.
- 80% stipend is defensible *because* your only real out-of-pocket research costs are presenting and a
  little training — but be ready to say exactly that if a reviewer asks.
- Answer the **"paid or unpaid," "other funding sources," and "need-based aid"** questions honestly —
  the stipend is fine, but the whole picture has to be consistent.
- **Don't** include tuition, loans, phone, utilities, rent/food, or a laptop.
- `[SWAP OK]` Conference alternatives if ISMB is too far/costly: **MLCB (Machine Learning in
  Computational Biology)** — very on-topic, often virtual/cheaper — or **NCUR (National Conference on
  Undergraduate Research)**. Course alternative: the free **Hugging Face Diffusion Models Course** (if
  you use the free one, move the $100 into the stipend or poster line).

### Copy-paste version for Q29 (itemized budget field) — matches the form's example format
```
Research stipend: $4,000

My research opportunity in the You Lab is unpaid and not for course credit, and I currently earn no income. This stipend compensates my research time at $15/hour for an approximately 20 hour/week commitment (about 265 hours total) during the 2026-2027 academic year (September 2026 to March 2027), allowing me to sustain that commitment to unpaid research rather than taking paid work.

Conference registration and travel: $800

Registration and travel to present a poster on my data-augmentation research at ISMB (Intelligent Systems for Molecular Biology), at its Undergraduate Research Symposium in 2027. This would be my first research presentation.

Research training course: $100

Enrollment in the DeepLearning.AI "GANs Specialization" on Coursera (about $49/month over roughly two months, winter 2027), which builds the generative image-modeling skills my ControlNet and diffusion project relies on.

Poster, figures, and research materials: $100

Poster printing, figure preparation, and small software or dataset costs during the project to produce and share my results.

Total: $5,000
```

---

## 5. STILL OPEN (fastest path to finishing)
Confirmed: undergrad ✓ · U.S. citizen ✓ · research unpaid, no course credit ✓ · no paid job ✓ ·
supervisor = Kinshuk Sahu ✓ · scope = data-augmentation project, growing into larger work ✓ ·
goal = learn comp bio deeply ✓.

Remaining:
1. **Kinshuk's email address** — needed for the supervisor letter (Q20). Give him a heads-up now.
2. **Optional polish (makes the app stronger):**
   - Name a real conference or course/workshop you'd target → makes the budget concrete.
   - One specific milestone for the data-augmentation project this year (a species, a benchmark, a
     figure) → sharpens Q16/Q22/Q23.
   - Your household income bracket (Q11) — I won't guess it; it's part of the need picture (the grant
     doesn't require hardship, so this is just honesty, not a hurdle).
3. **Personalize the personal statements (Q21–Q23) in your own words** — reviewers screen for
   AI-written text, so treat my drafts as scaffolding only.
