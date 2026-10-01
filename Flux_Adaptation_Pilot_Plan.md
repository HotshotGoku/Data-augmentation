# Flux adaptation pilot — plan & go/no-go
Sarthak, 2026-09-28 · for Kinshuk ("would adapting Flux be difficult? resolution an issue?")

## Bottom line
**Feasible, and the resolution is not the problem — the conditioning mechanism is the one real decision.** Flux runs on our 24 GB A5000s today (smoke-tested), and its native 1024 px actually *suits* branching (1001 px) better than SD1.5. The cost is a genuine engineering lift: a 12B model needs QLoRA to fine-tune, and — more important — our augmenter's "give me another replicate of *this* colony" conditioning is **not** a stock Flux ControlNet, so we must pick how to condition. This is a weeks-scale bet, not a config change. It's worth doing as a **bounded, one-domain pilot** because it's the clean test of Kinshuk's hypothesis that a stronger backbone preserves realism *and* diversity together (moves the frontier instead of sliding along it). If the pilot doesn't beat SD1.5's Pareto point on our frozen bench, we abandon — the backbone wasn't the lever.

## Why Flux at all (what the pilot tests)
Our bake-off mapped a realism↔diversity frontier with two levers (recipe, resolution) that trade one for the other. That coupling is partly an artifact of limited target data + small-LoRA capacity locking onto a prior — not a law. A better generator (Flux's flow-matching transformer, a stronger prior than SD1.5's U-Net) could push the whole frontier out. Our earlier VAE diagnostic showed the SD1.5 *VAE* is not the ceiling, so the case for Flux rests specifically on its **transformer/generator**. That's the single hypothesis this pilot exists to test.

## What we already know (hands-on smoke test, `flux_controlnet_smoke.py` / `flux_feasibility.md`)
- Env `flowmatch` (diffusers 0.31.0) has `FluxControlNetModel` / `FluxControlNetPipeline` out of the box. `FLUX.1-dev` (12B, gated — license already accepted on the lab HF account) + `InstantX/FLUX.1-dev-Controlnet-Canny`.
- Ran: Canny-of-source control, empty prompt, 512 px, 24 steps, sequential offload → **0.7 GB peak VRAM**, 151 s/img, load 65 s. Plumbing OK; output was generic texture — **because Canny throws away appearance and Flux has zero colony knowledge zero-shot.** That non-result is the whole point: a Canny ControlNet is the wrong conditioning for us.

## The one real decision: how to condition
Our task is *identity-preserving resampling*: given a source replicate, produce another plausible replicate of the same colony/condition, with diversity. Options, from most faithful/most work to least:

| Path | Mechanism | Fit for our task | Plumbing |
|---|---|---|---|
| **Trained Flux ControlNet (raw source, not Canny)** | Train a CN branch that takes the source *image* as the hint | Tightest structure/correspondence; direct port of the SD1.5 setup | High — train a CN branch against a 12B base |
| **Flux Kontext-dev + QLoRA** *(recommended to try first)* | In-context image regeneration; fine-tune on (source→sibling) pairs with a fixed instruction | Natural "another version of this image", preserves identity, resamples stochastic detail → good realism+diversity balance | Medium — LoRA on Kontext; recipe newer/less battle-tested |
| **Flux Redux / IP-Adapter** | Condition on a global SigLIP image embedding | Easy, diverse, but embedding is global → may drift spatial layout / lose correspondence | Low |
| **Canny/depth ControlNet (stock)** | Edge/depth control | ❌ discards appearance (smoke-test confirmed) | Lowest — but wrong |

**Recommendation:** if the goal is the *augmenter's core use* (realistic, diverse replicates of a condition), pilot **Kontext-dev + QLoRA** first; if a downstream task needs *tight* structure/correspondence (sim→exp, location-preserving exp→seed), pilot a **trained Flux ControlNet on the raw source** instead. Do **not** reuse the Canny CN — that's what made the smoke test look bad.

## Fine-tuning recipe (QLoRA on A5000, mirrors our PixCell LoRA precedent)
- **Quantize + freeze the base:** nf4/fp8 (bitsandbytes / optimum-quanto) → ~7–12 GB resident, fits 24 GB with headroom.
- **Train only adapters:** PEFT LoRA on the transformer's linear layers — start from the config that worked for PixCell (`r=16, alpha=32, target_modules="all-linear", dropout=0`, lr 1e-5), plus **gradient checkpointing** and an **8-bit Adam** optimizer. (For the trained-ControlNet path, train the CN branch with the base quantized+frozen.)
- **Data:** the exact replicate-pair JSONs we already build (`build_multiplexed_pairs.py` / `build_branching_pairs.py`) — source→sibling pairs, on-the-fly 90° rotation + flip.
- **Inference for eval/generation:** fp8 quant + `enable_model_cpu_offload` → seconds/image instead of the 151 s offload path.
- **Rough cost:** ~1–2 GPU-days for one domain (few-thousand pairs, a few epochs), slower per step than SD1.5 but unattended on youlab-gpu.

## Resolution — not a blocker, actually an asset
Flux is native 1024. Branching originals are 1001 px (we currently downsample to 256). A Flux pilot on branching should train/generate at **1024 from the `Exp_images` originals**, not upscaled small copies — this is the one place higher native resolution is a real win (and where SD1.5 is weakest).

## Staged pilot with kill criteria
1. **Inference sanity (½ day):** fp8-quant Flux, `enable_model_cpu_offload`, confirm seconds/image at 1024. → *kill if it won't fit fast enough even quantized.*
2. **Conditioning pick (½ day):** stand up the chosen path (Kontext-QLoRA or raw-source ControlNet) and overfit 1 condition to confirm it can reproduce a specific colony (not generic texture). → *kill if it can't be made to obey the source at all.*
3. **One-domain QLoRA (1–2 days):** fine-tune on multiplexed-256 **or** branching-1024 replicate pairs.
4. **Frozen-bench score (½ day):** run our existing `eval_metrics.py` (CMMD-336, realism-vs-sibling ÷ floor, diversity, copy_rate) on the same held-out reals → **apples-to-apples vs the SD1.5 numbers.**
5. **Go/no-go:** ship only if Flux beats SD1.5's Pareto point — ideally **higher realism at equal-or-higher diversity** (the frontier actually moved). If it just lands somewhere else on the same curve, the backbone isn't the lever and we stop.

## Evaluation = our current bench, unchanged
Same metrics, same held-out sets, same references as the bake-off, so the Flux row drops straight into the existing table. No new scoring code.

## Skeleton (key pieces — starting point, not tested end-to-end)
```python
# ponytail: skeleton — the real work is the conditioning path (Kontext vs trained CN) + QLoRA plumbing
import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel  # or Flux Kontext pipeline
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig  # nf4 quantization of the frozen base

# 1) load base quantized + frozen (nf4/fp8); adapters carry all the learning
#    (mirror pixcell1024_train_cn.py: build_pairs -> DDPM/flow-match loss on noisy target latents)
# 2) LoRA on the trainable branch — reuse the PixCell config that worked:
lora = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.0)
# transformer/controlnet = get_peft_model(<trainable branch>, lora)
# 3) enable gradient checkpointing + 8-bit Adam + accelerate offload
# 4) train on the SAME source->sibling replicate-pair JSONs we already build
# 5) inference: fp8 + enable_model_cpu_offload; score with eval_metrics.py
```

## Effort estimate
~1 focused week for a decisive one-domain pilot (most of it is the conditioning-path plumbing + QLoRA, not compute). If Kinshuk wants to run it, the conditioning-path choice is the first fork — I can stand up the training scaffold (dataset + QLoRA loop + the chosen conditioning) to hand off.
