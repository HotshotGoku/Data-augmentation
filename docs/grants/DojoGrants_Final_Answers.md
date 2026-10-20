# DojoGrants — Final Answers (copy-paste ready)

Before you submit: read each answer aloud once and swap a few word choices to your own phrasing so
it's unmistakably yours (the form screens for AI text). No em dashes used. After you submit, give
**Kinshuk** a heads-up that he'll get an automated Nucleate email with a letter-upload link (letter
due ~Sept 1) so it doesn't land in spam.

Already on the form and kept consistent: 20 hrs/week · July 2026 – May 2027 · unpaid, no course
credit, no income · supervisor Kinshuk Sahu.

---

## Research Opportunity — Scope (100 words max) · ~96 words

In Dr. Lingchong You's lab at Duke, I build machine-learning models of how bacteria self-organize into branching colony patterns (specifically Pseudomonas aeruginosa). My project is generative data augmentation: I fine-tune a ControlNet diffusion model to synthesize realistic experimental replicates, so pattern-prediction models can train on far more data than the wet lab produces. This year I want to improve generalization to species it wasn't trained on, add quantitative evaluation (SSIM and LPIPS) instead of judging by eye, and test whether synthetic data measurably improves a downstream predictor. It runs in PyTorch on Duke's GPU cluster.

---

## Personal Statement 1 — Project you took ownership of (1500 char) · 1,449

The project I've taken the most ownership of is DynaLab, a molecular-dynamics platform I've built mostly on my own over about two years. It grew out of research I did in high school, running protein simulations on a university supercomputer, and out of a frustration: those tools basically require an HPC cluster, a budget, and specialized training to touch. I wanted to put that capability in a browser.

So I built the whole surrounding system. A scale-to-zero cloud backend that wakes a simulation server on demand and stops it when idle, so it costs almost nothing at rest; a per-user quota system and auth; and a roughly 3,700-line analysis suite that turns raw simulation trajectories into structural insight, including free-energy estimates and rupture forces calibrated to real physical units.

The hard part wasn't any single piece. It was being the only person holding both halves at once, the biophysics and the systems engineering, and making them meet in the middle. I had to understand the science well enough to know which outputs mattered, and the infrastructure well enough to serve them cheaply and reliably. A lot of it I learned by breaking things first.

What I took from it is that follow-through is the real skill. Plenty of people can prototype something; finishing it, validating it against the literature, and keeping it running for two years is harder and rarer. It's also what convinced me I actually want to do research.

---

## Personal Statement 2 — Research question you're most interested in (1500 char) · 1,358

The problem I keep coming back to is that machine learning in biology almost never has enough data. In the You Lab I run into it directly: every bacterial colony pattern we'd want to train on takes real bench time to grow and image, so we can't produce the tens of thousands of examples a modern model actually wants. You end up with powerful methods starved of the thing that makes them work.

The question I'm most interested in is whether a generative model can learn the natural variation in biological data well enough to create new, realistic examples that make a downstream model genuinely better, not just examples that look convincing. What pulls me in is that it sits between physics and deep learning. Reaction-diffusion simulations give you clean, controllable "ground truth," and diffusion models add back the messiness of real experiments. It also has a failure mode I can actually chase: the augmenter does well on species it has seen and breaks on ones it hasn't, which tells me it's still memorizing variation rather than understanding it.

I care about this because if it works here, it isn't really about bacteria. Almost every part of biology where experiments are slow and data is scarce has the same bottleneck. Getting a model to honestly expand a small dataset feels like one of the more useful things I could be working on right now.

---

## Personal Statement 3 — Next goals + how DojoGrants helps (2000 char) · 1,840

My most honest goal right now is to learn as much computational biology as I can, and this project is how I'm doing it. I'm a sophomore, and I've spent the last few years building at the edge of biology and software: a molecular-dynamics platform, wet-lab and imaging work, and now generative modeling. The You Lab is where I get to go deep on the machine-learning side of it.

Concretely, I want to use this year to go from setting the data-augmentation pipeline up to actually owning it: getting the generative augmenter to work on bacterial species it currently fails on, replacing "looks right" with real quantitative evaluation, and building toward the larger lab projects my mentor and I plan to grow into. I'd also like a first real research output, a poster or short write-up I can present, and a clearer answer to whether I want to do computational-biology research long term. I'm fairly sure I do, but I'd rather earn that certainty by doing the work than assume it.

DojoGrants would help on both the practical and the developmental side. The funding takes the friction off an unpaid project: it lets me commit serious weekly hours to research instead of taking a paid job, present my work at ISMB's undergraduate symposium for the first time, and pick up formal skills faster through a focused course. The program itself is almost built for where I am. Mentorship from PhDs and industry members would help me decide what to learn next; the presentation practice and Annual Symposium would give me my first research talk; the founder and industry sessions plus the Nucleate network would show me how computational biology becomes real tools and companies; and the cohort gives me people to learn alongside. I'm early in this, and DojoGrants is the kind of push that would move me faster and more seriously than I could on my own.

---

## Optional — Anything else (2500 char) · 1,321

For a bit of context on where I'm coming from: I'm a sophomore, but I've been doing research since high school. I taught myself to code at sixteen so I could work in a computational-biology lab at the University of Chicago, running protein simulations on their supercomputer, and I've been building at the intersection of biology and software ever since. A lot of what's on my resume, a molecular-dynamics platform, computer-vision systems, wet-lab imaging work, came from following that thread on my own, usually by starting things before I fully knew how to finish them and working out the rest as I went.

I mention it because the You Lab project is the most focused version of that same instinct. It's the first time I get to go deep on modern generative ML inside real biology research, with a mentor and a lab pushing me, instead of teaching myself in isolation. The grant would let me commit to it seriously at a point where I'm early enough that the momentum genuinely matters.

I don't come from a research family and I don't have all of this mapped out. I've mostly gotten here by being willing to take on things that were a little beyond me and staying with them until they worked. That's exactly how I plan to approach this project, and I think it's the part of my background that a resume doesn't quite show.

---

## Funding Request — how funding helps + itemized budget (one field)

My research in the You Lab is unpaid, not for course credit, and I currently earn no income. I plan to commit about 20 hours a week to it during the 2026-2027 academic year. My compute is provided free through the Duke Compute Cluster, so this grant would go to two things: sustaining the time I can put into research, and sharing the results. Every item below ties to research participation or output, not general living costs.

Research stipend: $4,000

My research is unpaid and not for course credit, and I earn no income. This is a partial stipend at $15/hour that offsets roughly 265 hours of my ~20 hour/week commitment during the 2026-2027 academic year (about September 2026 to May 2027), so I can sustain that time on research instead of taking a paid job.

Conference registration and travel: $800

Registration and travel to present a poster on my data-augmentation work at ISMB (Intelligent Systems for Molecular Biology), at its undergraduate research symposium in 2027. This would be my first research presentation.

Research training course: $100

A generative-modeling course to build the foundations my ControlNet and diffusion work relies on (for example, DeepLearning.AI's GANs Specialization on Coursera, about $49/month for roughly two months).

Poster, figures, and research materials: $100

Poster printing, figure preparation, and small software or dataset costs to produce and share my results.

Total: $5,000
