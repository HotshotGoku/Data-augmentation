"""Flux pilot - Stage 1: quantized inference sanity at 1024, validating the fp8 base for QLoRA.
diffusers 0.31 has no native BnB quant, so we fp8-quantize the transformer + T5 with optimum-quanto
(load bf16 -> quantize -> freeze), then generate at 1024 and report peak VRAM + speed. If this fits
fast, the QLoRA fine-tune (base frozen+fp8, LoRA trainable) is feasible on the 24GB A5000.
Env: FLUX_BASE, HW[1024], STEPS[28], OFFLOAD[model|none]."""
import os, time, torch
from diffusers import FluxPipeline
from optimum.quanto import quantize, freeze, qfloat8

BASE = os.environ.get("FLUX_BASE", "black-forest-labs/FLUX.1-dev")
HW = int(os.environ.get("HW", "1024"))
STEPS = int(os.environ.get("STEPS", "28"))
OFFLOAD = os.environ.get("OFFLOAD", "model")   # model=whole-module offload; none=keep resident

t0 = time.time()
pipe = FluxPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16)
print(f"[load] pipeline bf16 in {time.time()-t0:.0f}s", flush=True)

for name, mod in [("transformer", pipe.transformer), ("text_encoder_2(T5)", pipe.text_encoder_2)]:
    t = time.time(); quantize(mod, weights=qfloat8); freeze(mod)
    print(f"[quant] {name} -> fp8 in {time.time()-t:.0f}s", flush=True)

if OFFLOAD == "model":
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")

torch.cuda.reset_peak_memory_stats()
t1 = time.time()
img = pipe(prompt="a bacterial colony on agar, top-down microscopy", height=HW, width=HW,
           num_inference_steps=STEPS, guidance_scale=3.5,
           generator=torch.Generator("cpu").manual_seed(0)).images[0]
dt = time.time() - t1
os.makedirs("flux_quant_out", exist_ok=True)
img.save(f"flux_quant_out/gen_{HW}.png")
print(f"[gen] {HW}px {STEPS} steps in {dt:.0f}s | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB "
      f"| offload={OFFLOAD} -> flux_quant_out/gen_{HW}.png", flush=True)
