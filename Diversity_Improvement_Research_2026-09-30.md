# Improving sample diversity in our Flux ControlNet augmenter — research review
Sarthak, 2026-09-30. Deep-research synthesis (21 sources, 25 claims adversarially verified). Context: Flux.1-dev ControlNet replicate augmenter, realism 0.523 / CMMD 6.5 (both beat SD1.5) but diversity (intra-source pairwise LPIPS) low at 0.22. Goal: raise diversity, hold realism/CMMD.

## TL;DR
The realism↔diversity tradeoff **can** be broken — but the classic "lower CFG" lever is **weak for us** because Flux.1-dev is guidance-distilled. The best-supported levers that don't depend on a live CFG knob are: **(1) CADS** — annealed noise on the conditioning (cheapest), and **(3) Feature Self-Guidance** — feature-space repulsion, the only method validated *on Flux* — with **(2) Particle Guidance** (batch repulsion) in between. All are **inference-time, no retraining**.

## The techniques (verified)

### 1. CADS — Condition-Annealed Diffusion Sampling  ★ cheapest, strongest evidence
- **What:** adds scheduled, monotonically-decreasing Gaussian noise to the *conditioning* during sampling: `ŷ = √γ(t)·y + s·√(1−γ(t))·n`, piecewise-linear `γ` (clean early→noisy... actually noisy early, clean late as t:1→0). Corrupts conditioning early (model explores) and restores it late (alignment preserved). A `ψ` rescaling knob trades stability↔diversity (`ψ=1` default, lower for more diversity).
- **Inference-only**, any sampler, condition-agnostic → transplants to our ControlNet conditioning embedding (~15–20 lines).
- **Effect — the "improve both" result:** ImageNet-256 FID 20.8→9.5, Recall 0.32→0.62; DeepFashion pose→image FID 16.4→7.7, Recall 0.02→0.48, Vendi 1.04→2.31, condition alignment essentially unchanged. *Caveat:* "both improve" leans on FID (which mixes quality+diversity); pure per-sample Precision drops slightly (0.90→0.77).
- Cite: arXiv 2310.17347 (ICLR 2024); impl github.com/v0xie/sd-webui-cads.
- **Key tuning caveat for us:** the 0.25 noise default is tuned for *text*; the paper's spatial/pose conditioning needed **3–4× larger noise** (s≈0.8–1.0). So for our image conditioning, expect to tune noise **up** and check CMMD empirically — a refuted-but-real warning is that text-tuned conditioning-noise can transfer poorly to image/ControlNet conditioning.

### 2. Particle Guidance — batch repulsion
- **What:** sample a batch jointly (not i.i.d.); add a repulsion term `−α_t·∇ Σ_j k(x_i,x_j)` (e.g. RBF) that pushes the batch apart. Needs a batch ≥2 for the **same** source — a natural fit for our per-source replicate generation.
- **Inference-only**, any frozen model. Lower in-batch similarity at matched CLIP/quality (not just a lowered-CFG artifact).
- **Caveat:** fixed-potential PG is **not proven to preserve the marginal** (a CMMD risk); follow-ups report artifacts. Its lever is partly CFG-coupled (weakened by Flux distillation).
- Cite: arXiv 2310.13102 (ICLR 2024).

### 3. Feature Self-Guidance ("Don't Settle at the Mode!")  ★ most on-point
- **What:** diagnoses diversity collapse *specifically in FLUX.1-dev / FLUX.1-Depth-dev / FLUX-Kontext* under identical conditioning; fixes it training-free by **dispersing internal MMDiT features** across a generation batch + a **manifold-projection** step to stay on-distribution.
- **Inference-only**, **CFG-independent** (doesn't need the distilled guidance knob) → safest mechanism for our setup. Reports diversity gains with preserved fidelity on text-, **depth-**, and **reference-image** generation — i.e. our exact ControlNet / image-to-image regime.
- **Caveats:** single source (ECCV 2026, couldn't verify exact fidelity tables); **batch-level** (must generate the N replicates for a source together); "plug-and-play" means implementing **feature hooks into the transformer internals** (a specific block over a timestep window), not a config flag — higher engineering cost than CADS.
- Cite: arXiv 2606.27371 (ECCV 2026).

### Deprioritized
- **Lower CFG scale:** genuine lever in general, but Flux.1-dev is **guidance-distilled** → weak/removed. Lowest priority. (Cite: Ho & Salimans 2207.12598.)
- **Learned guidance schedule** (arXiv 2506.24108): needs training a scheduler, and its dual-metric + flow-matching claims were **refuted** in verification. Skip.
- **`controlnet_conditioning_scale`:** not a paper finding but a trivial one-line inference knob — lower it (1.0→0.7) = weaker conditioning = more diversity, at some alignment cost. Free first probe.

## Prioritized, cheapest-first plan for our setup
1. **`controlnet_conditioning_scale` sweep** (one-line, minutes): 1.0 → {0.85, 0.7, 0.5}. Free probe of the diversity/alignment curve.
2. **CADS on the ControlNet conditioning** (~20 lines, inference-only): noise-scale sweep (start ~0.5–1.0, *above* the text default), `ψ` knob to hold CMMD. Best evidence, cheapest principled option.
3. **Particle Guidance** across the per-source replicate batch (we already generate N/source — make them a jointly-repelled batch). Watch CMMD.
4. **Feature Self-Guidance** — most on-point + Flux-validated + CFG-independent, but needs MMDiT feature hooks; do it if 1–3 don't get us there.

## Honest caveats (must-read before trusting any of this)
- **Metric transfer:** none of the sources measured *our* metrics (intra-source pairwise LPIPS, CMMD, LPIPS-to-nearest-real). Their evidence is Recall/Vendi/FID — directional, not guaranteed. **Re-measure CMMD + diversity + realism after every method** on our frozen bench.
- **Noise-scale is unknown for microscopy image conditioning** — the text 0.25 and pose 0.8–1.0 values only bracket it. Tune empirically.
- **Training-time branch is under-evidenced:** conditioning dropout / contrastive-flow-matching (arXiv 2506.05350) / SubFlow surfaced by name but *no training-time technique survived verification*. If inference-time levers cap out, these are the retraining options to investigate next.
