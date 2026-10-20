# Flux ControlNet — feasibility on our hardware (Track B)

Hands-on smoke test result + effort assessment, to decide whether Flux is a viable next-gen backbone
for the replicate augmenter (vs Kinshuk's state-space demo).

## What was tested
- **Env:** `flowmatch` (diffusers 0.31.0, torch 2.4.1+cu121, accelerate 1.14) — has full Flux ControlNet
  support out of the box (`FluxControlNetModel`, `FluxControlNetPipeline`, `FluxTransformer2DModel`).
- **Models:** `black-forest-labs/FLUX.1-dev` (12B, gated — license now accepted on the lab HF account)
  + `InstantX/FLUX.1-dev-Controlnet-Canny`. Note: `FLUX.1-schnell` is **also gated**, and the public
  ControlNets are all trained for `FLUX.1-dev`, so dev is the right target.
- **Run:** control = Canny of a multiplexed source, empty prompt, 512×512, 24 steps, sequential CPU offload.

## Result — it RUNS on a 24 GB A5000
| | |
|---|---|
| Load (Flux 12B + T5 + ControlNet, bf16) | 65 s |
| Generate (512px, 24 steps, sequential offload) | 151 s / image |
| **Peak VRAM** | **0.7 GB** |
| Output | plumbing OK; zero-shot image is a generic texture (expected — Flux has no colony knowledge, empty prompt) |

**Takeaways**
1. **VRAM is not a blocker for inference.** Sequential CPU offload keeps peak GPU memory at ~0.7 GB — there's enormous headroom on the 24 GB card. The cost is speed: 151 s/image (offload shuttles each layer CPU↔GPU).
2. **Faster inference is available by trading that headroom for VRAM:** `enable_model_cpu_offload` (whole-module) or, better, **fp8/nf4 quantization** (optimum-quanto / bitsandbytes) keeps the ~12B transformer resident (~12 GB fp8) → seconds/image instead of minutes, comfortably within 24 GB.
3. **Fine-tuning is the real question, and it's a genuine lift.** A full fine-tune of the 12B transformer will not fit on 24 GB. The feasible path is **QLoRA-style**: nf4/fp8-quantized frozen base + LoRA adapters (or train only the ControlNet branch with the base quantized+frozen) + gradient checkpointing + 8-bit optimizer + offload. Doable on the A5000, but slower per step and meaningfully more plumbing than our SD1.5 ControlNet fine-tunes.

## Recommendation
Flux ControlNet is **feasible on our hardware** for both inference (via offload or, better, fp8 quantization) and fine-tuning (via QLoRA) — no hard blocker, but fine-tuning is a real engineering investment.

**Given Kinshuk is building a state-space demo**, the sensible move is to *decide which backbone gets the fine-tune effort* before building it, rather than fine-tune both. If we choose Flux:
- Cheap first step: fp8-quantize + `enable_model_cpu_offload` for fast inference, then **QLoRA fine-tune a Flux ControlNet on the multiplexed/branching replicate pairs**, and evaluate with the **new CMMD + per-sample suite** — giving an apples-to-apples comparison against the SD1.5 numbers.
- Because our diagnostic showed the SD1.5 *VAE* isn't the ceiling, the case for Flux rests on its **flow-matching transformer** (a better generator) closing the generative gap — that's the hypothesis a Flux QLoRA fine-tune would test.

**Artifacts:** `flux_controlnet_smoke.py`, `flux_smoke_sa603.sh`, outputs in `flux_smoke_out/` (source, control_canny, flux_gen).
