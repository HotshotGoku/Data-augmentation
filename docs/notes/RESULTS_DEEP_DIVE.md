# Augmenter Project — Deep-Dive on Everything We Did (Aug 25–27, 2026)

*A complete, honest account for Sarthak. Written to be read carefully before presenting to Kinshuk.*
*All code + results committed to branch `sa603/experiment-tracking-and-sweeps` on github.com/HotshotGoku/Data-augmentation.*

---

## 0. The one-paragraph summary

We took the replicate-to-replicate augmenter (which works in-distribution but fails out-of-distribution) and ran five experiments to (a) fix the OOD failure and (b) test whether its synthetic data actually helps downstream models. **Result: fine-tuning on a new species reliably fixes OOD (FID 359→~85; realism 2.5×→1.16× the real-vs-real baseline); layer-freezing helps modestly; inference-parameter tuning does not help OOD at all.** But **naive downstream augmentation did not help** — it hurt the sim→exp predictor, and our augmenter failed a location-preservation check for exp→seed. The unifying reason for every negative: the augmenter is *faithful only in-distribution*; out-of-distribution it imposes its trained branching prior and ignores the input. The practical takeaway: the augmenter must be adapted (fine-tuned) to a target distribution before it can help there.

---

## 1. What the augmenter is, and the problem

**The model.** A ControlNet + Stable Diffusion 1.5 system. You give it one experimental bacterial-colony image (a "source" replicate) and it generates new synthetic replicates of that same colony. The goal is to beat experimental data scarcity — generate many plausible replicates from few real ones.

**Your trained model.** Checkpoint `version_52034860` — trained from scratch, ~3 days, 375,749 steps, on 3 experimental datasets (all P. aeruginosa branching morphology).

**What was already known before this session:**
- **In-distribution: it works.** Generated replicates sit ~1.3–1.4× as far (in LPIPS perceptual distance) from real replicates as real replicates sit from each other — i.e., nearly as realistic as real, and diverse (not mode-collapsed).
- **Out-of-distribution: it fails.** On a new species (Kristen's 2-species fluorescence co-culture), it ignores the input and paints its trained grayscale branching pattern. This OOD failure is the central problem of the project.

**A metric primer (so the numbers below make sense):**
- **LPIPS** = a learned perceptual distance between two images. Lower = more similar.
- **realism_vs_sibling** = LPIPS from a generated image to the *other real replicates* of that colony. Lower = the fake looks more like a real replicate.
- **baseline_real_vs_real** = LPIPS between two *real* replicates. This is the yardstick — how different real replicates naturally are. If realism_vs_sibling ≈ baseline, the generated image is as close to real as reals are to each other (ratio ≈ 1.0× = ideal).
- **diversity** = LPIPS between the model's own two outputs for one input. Higher = more varied (guards against mode collapse).
- **FID** = distance between the feature *distributions* of generated vs real image sets. Lower = better. **Caveat: FID is noisy on small sets** (our OOD test has ~12 generated / ~48 real), so we treat it as indicative and lean on realism÷baseline.
- **SSIM** (downstream) = structural similarity, higher = better. Used for the sim→exp predictor.

---

## 2. Infrastructure we had to build first (the enabling work)

Before any experiment could run reliably, four things had to be fixed. These matter because they're why the results are trustworthy and reproducible.

1. **Disk unblocked.** The lab's `/hpc/group/youlab` shared space was **99% full (46 TB, ~0.7 TB free)** — this is what had killed your earlier fine-tune (its checkpoint truncated mid-write). We freed space by *scoring-then-deleting* redundant checkpoints and mirrored visualizations, and adopted a rule: never hold more than one run's checkpoints; score, keep the best, delete the rest. **This is a lab-wide problem worth flagging to Kinshuk** — it will break anyone's large writes.

2. **Environment fix (important gotcha).** Installing modern `wandb` silently upgraded `pydantic` to v2, which cascaded and broke `transformers`/`torchmetrics`/`pytorch_lightning` — the whole training stack. Symptom: a scoring job "COMPLETED" but every checkpoint logged `infer FAILED` and produced empty results. Fix: pin `wandb 0.16` (predates the pydantic-2 requirement) and **decouple W&B from compute jobs** — every job writes durable file records; a separate script syncs to W&B. So W&B can never crash a training run.

3. **Experiment tracking (your "document everything" ask).** Every run now writes a `manifest.json` (all hyperparameters + random seed + git commit + SLURM job id + timestamp) and appends a row to a central `runs.tsv`; results also sync to a Weights & Biases dashboard. Plus per-experiment metric summaries and visual reports.

4. **Layer-freezing implementation (a real subtlety).** The naive way to freeze layers (`requires_grad=False`) *crashes* this model, because it uses gradient checkpointing, which re-runs each block's backward and errors if any parameter doesn't require grad. The correct fix: keep all gradients flowing but **exclude the frozen parameters from the optimizer**, so they get gradients (no crash) but are never updated (frozen in effect). Verified working.

---

## 3. Experiment 1 — Fine-tuning fixes the OOD failure  ✅ (the key positive result)

**What we did.** Fine-tuned your base model on the 2-species (KL) training split, evaluated on a *zero-leakage held-out* split of that same species. Baseline = the un-fine-tuned model scored on the same OOD test.

**Results (OOD 2-species; baseline_real_vs_real = 0.32; lower realism = better):**

| Model | FID | realism_vs_sibling | ÷ baseline |
|---|---|---|---|
| Baseline (no fine-tuning) | 359 | 0.820 | **2.54×** |
| Fine-tuned (best) | ~85–90 | ~0.374 | **1.16×** |

**Interpretation.** Fine-tuning is *the* lever for OOD. It takes the model from "2.5× worse than real replicates differ from each other" down to "1.16× — nearly indistinguishable from real replicate variation." FID drops ~4×. Diversity stays healthy (~0.33), so it is not achieving realism by mode-collapsing. Visually (in `results_report/`), the baseline paints grayscale branching on the fluorescence input; the fine-tuned model produces correct green+red fluorescence colonies matching the real ones.

**Caveats (be precise with Kinshuk):**
- This is **adaptation, not zero-shot.** The model was fine-tuned on 2-species data and tested on held-out 2-species data. It proves the augmenter can be *efficiently adapted* to a new species with some of its data. It does **not** prove it generalizes to a species it has never seen any data from — that remains open.
- The test set is small, so FID is noisy; we rely on realism÷baseline, which is more stable but still has variance.

---

## 4. Experiment 2 — Layer freezing helps (modestly)  ✅

**What we did.** Kinshuk suggested freezing some ControlNet layers during fine-tuning so we "don't jumble everything." We implemented three modes — `none` (full fine-tune), `hint` (freeze the input-hint encoder), `shallow` (freeze hint + the early feature-extraction blocks) — and compared, then swept learning rate.

**Results (OOD 2-species, best epoch of each):**

| Config | FID | realism | ÷ baseline |
|---|---|---|---|
| Fine-tune, no freezing (`none`) | 98.8 | 0.387 | 1.20× |
| Fine-tune, freeze shallow | **89.6** | **0.374** | **1.16×** |

Learning-rate sweep on `shallow` (×6 epochs): `lr=5e-6` and `lr=1e-5` both good (FID ~85–90); `lr=1e-6` too slow. Everything **plateaus around 1.16–1.20× / FID ~85** — more tuning gives diminishing returns.

**Interpretation.** Freezing the early layers does help — better FID and realism than full fine-tuning, and it adapts more stably (its curve improves monotonically). The effect is real but *modest*. Kinshuk's intuition was correct.

**Confidence note (a genuine strength):** the seeded `lr=5e-6, shallow` run *reproduced its predecessor exactly* (FID 89.49 vs 89.64; realism 0.3741 vs 0.3741) across two independent SLURM jobs. That's strong evidence the whole pipeline (seeding, data, scoring) is trustworthy and the freezing effect isn't noise.

---

## 5. Experiment 3 — Inference-parameter tuning does NOT fix OOD  ✅ (informative negative)

**What we did.** Kinshuk asked to try raising the guidance `scale` from 9.0 to 15.1 at inference. We swept the full grid: `scale ∈ {9, 12, 15.1}` × control `strength ∈ {1.0, 1.5, 2.0}` (9 configs), on the base checkpoint, scored on both OOD and in-domain.

**A key scientific insight going in:** the "scale" knob is classifier-free guidance (CFG), which amplifies the *text-prompt* contrast. But our prompts are empty and the control image is present in both guidance branches — so CFG mostly changes sharpness, not how much the model obeys the input. The knob that governs "listen to the input shape" is control `strength`. So we swept both.

**Results:**
- **OOD:** every one of the 9 configs stayed at realism **2.5–2.8× baseline** — none fixed it. Raising `scale` barely moved realism (0.82→0.82). Raising `strength` made it **worse** (0.82→0.91).
- **In-domain:** the current default (`scale=9.0, strength=1.0`) is already near-optimal (FID 82.7). Raising either *hurt* (FID up to 134 at strength 2.0; up to 95 at scale 15.1).

**Interpretation.** You cannot fix OOD with inference knobs — the failure lives in the *learned weights*, not the sampling parameters. Cranking `strength` is actively counterproductive because it amplifies the *wrong* trained prior harder onto the new input. And Kinshuk's `15.1` specifically would slightly *degrade* in-domain quality. This experiment cleanly rules out the whole "cheap inference fix" class and points to fine-tuning — which is exactly what Experiment 1 confirms works.

---

## 6. Experiment 4 — Downstream utility (sim→exp): naive augmentation HURTS  ⚠️ (negative, with an owned design flaw)

**Why this experiment.** Everything above shows the augmenter *makes realistic pictures.* The claim the lab actually needs is that its synthetic data is *useful* — that a downstream model trained on real+synthetic beats one trained on real alone. The lab's flagship downstream model is the **sim→exp predictor** (given a simulation image, predict the experimental colony).

**What we did.** Two trainings, identical except for data: **A** = real-only (5,000 sim→exp pairs from 50 colonies), **B** = real + 1,000 synthetic replicates from our augmenter. Both evaluated on the same held-out *real* test set (96 images), same seed. Metrics: SSIM (↑) and LPIPS (↓).

**Results:**

| Condition | SSIM ↑ | LPIPS ↓ |
|---|---|---|
| A — real-only | **0.392** | **0.378** |
| B — real + synthetic | 0.195 | 0.490 |

Adding synthetic data **halved SSIM** and worsened LPIPS. The prediction images confirm it: B collapsed to a generic radial "sunburst" that ignores each specific simulation.

**Mechanism — and a design flaw I own.** The sim→exp task needs *matched* pairs: this specific simulation → the colony it actually produces. But the augmenter generates *diverse* replicates, and I paired all 20 synthetic replicates of a colony with that colony's *single* rotation-0 simulation. So B's data effectively said "this one simulation → 20 different colonies," injecting one-to-many noise into a correspondence mapping. The model coped by ignoring the simulation and reverting to a generic prior.

**What this does and doesn't prove.** It proves that *this naive way of injecting replicate-augmentation into a correspondence task hurts.* It does **not** prove synthetic data is useless for sim→exp — the injection broke the pairing by construction. A correspondence-preserving injection might behave differently. (This is the honest framing; don't let it be read as "synthetic data doesn't work.")

---

## 7. Experiment 5 — Downstream utility (exp→seed): stopped at a pre-flight gate  ⚠️

**Why exp→seed.** It's the *inverse* model (experimental colony → the 32×32 map of initial seed locations). Here experimental images are *inputs*, and many replicates legitimately map to *one* seed — so replicate-augmentation is correspondence-preserving (the opposite of the sim→exp trap). This should be the setting where the augmenter *can* help.

**A discovery worth telling Kinshuk:** the exp→seed model (Fig 6 in the paper) is **already trained 100% on synthetic data** — 90,000 sim→exp *outputs*, generated precisely because real experimental data is scarce. So the lab's pipeline is *itself* a synthetic-augmentation success story, just using in-distribution synthetic data.

**The pre-flight gate.** The whole experiment depends on the augmenter *preserving colony/seed locations* in its replicates (a replicate must share its seed's layout). Before spending ~a day of 4-GPU compute, I generated a few replicates and looked. **It failed the gate:** the augmenter replaced the scattered multi-colony input dishes with a dense radial branch — locations, counts, and layout not preserved. Evidence: `model_results/expseed_gate/expseed_location_gate.png`.

**Why it failed:** these inputs (the `Selected_v4` scattered multi-colony dishes) are *out-of-distribution* for the augmenter (it was trained on single dominant branching colonies), so it hallucinated its prior — the same failure mode again. **We stopped rather than train on a broken premise.** Caveat: "stopped," not "impossible" — if the augmenter were first fine-tuned on this layout, it might preserve locations.

---

## 8. The unified finding (the through-line for Kinshuk)

Five experiments, one consistent mechanism:

| Setting | Outcome |
|---|---|
| In-distribution replicates | ✅ Faithful, realistic, diverse |
| OOD species (2-species) | ❌ Imposes branching prior → **fine-tuning fixes it** |
| Inference-param tuning for OOD | ❌ No knob fixes it (strength makes it worse) |
| Downstream sim→exp augmentation | ❌ Naive injection breaks correspondence → hurts |
| Downstream exp→seed augmentation | ❌ Doesn't preserve layout (OOD inputs) → gated out |

**The augmenter is faithful only in-distribution.** Out-of-distribution — whether a new *species* or a new *spatial layout* — it imposes its trained dense-radial-branching prior and ignores the input. Therefore it is not yet a plug-in downstream augmenter for tasks needing input-faithfulness, **unless first fine-tuned to the target distribution** (which we proved works for OOD species).

**A reusable methodological lesson:** gate augmentation-utility experiments with a cheap visual/correspondence check *before* training. It caught the sim→exp flaw in hindsight and saved a day of compute on exp→seed.

---

## 9. Flow matching (Kinshuk's point 4) — written go/no-go

Full review in `flow_matching_review.md`. Short version: flow matching (as in Stable Diffusion 3 / FLUX) is a genuinely stronger *backbone* — better prior, faster/higher-fidelity sampling — and ControlNet analogs exist for it. **But it is not a targeted fix for our OOD problem**, which is about distribution coverage, not the diffusion-vs-flow objective. Recommendation: **not now.** Finish squeezing the current model; if a backbone upgrade is pursued, run a bounded SD3-medium pilot (VRAM-realistic on our A5000s) as a controlled comparison, *after* the current levers are exhausted. It's a weeks-scale bet, not a config change.

---

## 10. Honest limitations & things to double-check before over-claiming

- **"Fixed OOD" = adaptation, not zero-shot.** Fine-tuning was tested on held-out data of the *same* species it was fine-tuned on. True zero-shot (a species with zero training data) is still open.
- **Small test sets → noisy FID.** We de-emphasize FID and lean on realism÷baseline + diversity; even those have variance on ~50-image tests.
- **The sim→exp negative has a design flaw (owned).** It shows *naive* injection hurts, not that synthetic data can't help that task.
- **The exp→seed result is a gate, not a trained comparison.** It shows the current augmenter isn't location-faithful on those inputs; a fine-tuned augmenter might be.
- **Freezing's benefit is modest and mostly single-seed** (except the lr5e-6 point, which reproduced exactly). Worth another seed or two before calling it definitive.
- **Everything is on P. aeruginosa-type morphology.** Generalization claims should stay scoped to that.

---

## 11. What I'd want Kinshuk to weigh in on

1. **Is "efficient adaptation to a new species" a satisfying result on its own,** or is the bar "true zero-shot / downstream utility"? This sets the direction.
2. **Downstream utility redesign:** worth doing a correspondence-preserving sim→exp injection, or fine-tune-then-augment (adapt the augmenter to the downstream domain first, then test utility)?
3. **Backbone:** any appetite for the SD3-medium flow-matching pilot, or stay with SD1.5?
4. **The lab-wide disk issue** (youlab group 99% full) — needs a cleanup or a request to research computing.

---

*Where to look:* metrics in `sweep_eval_2sp/summary.tsv` (fine-tune) and `infer_sweep_out/summary.tsv` (inference); visual reports in `model_results/results_report/` (OOD fine-tune + inference grids), `model_results/downstream_report/` (sim→exp A vs B), `model_results/expseed_gate/` (the gate). Run records in `experiments/runs.tsv`.
