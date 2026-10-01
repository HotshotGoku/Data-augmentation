"""Flux pilot - ControlNet fine-tune (QLoRA-style): base transformer fp8-frozen (quanto), a small
FluxControlNet trainable, flow-matching loss, bnb 8-bit Adam, grad checkpointing. Reuses our
replicate-pair JSONs (source replicate -> target replicate). Grounded in the diffusers
train_controlnet_flux.py recipe (flow matching: t=sigmoid(randn), noisy=(1-t)x0+t*noise, target=noise-x0).
Env: DATA_JSON(req), RES[512], MAX_PAIRS[0=all], MAX_STEPS[2000], LR[1e-5], CN_LAYERS[4], ACCUM[4],
     SAVE_EVERY[500], OUT, GRAD_CKPT[1]."""
import os, json, time, random
import cv2, torch
import torch.nn.functional as F
from diffusers import FluxPipeline, FluxControlNetModel
from optimum.quanto import quantize, freeze, qfloat8
import bitsandbytes as bnb

BASE = os.environ.get("FLUX_BASE", "black-forest-labs/FLUX.1-dev")
DATA_JSON = os.environ["DATA_JSON"]
RES = int(os.environ.get("RES", "512"))
MAX_PAIRS = int(os.environ.get("MAX_PAIRS", "0"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "2000"))
LR = float(os.environ.get("LR", "1e-5"))
CN_LAYERS = int(os.environ.get("CN_LAYERS", "4"))
ACCUM = int(os.environ.get("ACCUM", "4"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "500"))
GRAD_CKPT = os.environ.get("GRAD_CKPT", "1") == "1"
OUT = os.environ.get("OUT", "flux_cn_runs/pilot")
os.makedirs(OUT, exist_ok=True)
DEV, DT = "cuda", torch.bfloat16
torch.manual_seed(0); random.seed(0)

def _pack(l, B, C, H, W):
    return l.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 2) * (W // 2), C * 4)
def _img_ids(H, W):
    ids = torch.zeros(H // 2, W // 2, 3)
    ids[..., 1] += torch.arange(H // 2)[:, None]; ids[..., 2] += torch.arange(W // 2)[None, :]
    return ids.reshape((H // 2) * (W // 2), 3).to(DEV, DT)

# --- load base pipeline, precompute the (fixed) empty-prompt embeds, then drop the text encoders ---
pipe = FluxPipeline.from_pretrained(BASE, torch_dtype=DT)
pipe.text_encoder.to(DEV); pipe.text_encoder_2.to(DEV)
with torch.no_grad():
    pe, ppe, tids = pipe.encode_prompt(prompt="", prompt_2="", device=DEV, num_images_per_prompt=1, max_sequence_length=512)
pe, ppe, tids = pe.to(DT), ppe.to(DT), tids.to(DT)
del pipe.text_encoder, pipe.text_encoder_2; torch.cuda.empty_cache()
print(f"[flux-cn] empty embeds pe{tuple(pe.shape)} ppe{tuple(ppe.shape)} tids{tuple(tids.shape)}", flush=True)

vae = pipe.vae.to(DEV).eval()
for p in vae.parameters(): p.requires_grad_(False)
transformer = pipe.transformer
controlnet = FluxControlNetModel.from_transformer(transformer, num_layers=CN_LAYERS, num_single_layers=0)  # small CN from base
quantize(transformer, weights=qfloat8); freeze(transformer)
transformer.to(DEV).eval()
for p in transformer.parameters(): p.requires_grad_(False)
controlnet.to(DEV, DT).train()
if GRAD_CKPT:
    transformer.enable_gradient_checkpointing(); controlnet.enable_gradient_checkpointing()
opt = bnb.optim.Adam8bit(controlnet.parameters(), lr=LR)
print(f"[flux-cn] controlnet trainable {sum(p.numel() for p in controlnet.parameters() if p.requires_grad)/1e6:.0f}M | CN_LAYERS={CN_LAYERS}", flush=True)

pairs = [json.loads(l) for l in open(DATA_JSON)]
if MAX_PAIRS: pairs = pairs[:MAX_PAIRS]
def load_img(p):
    im = cv2.resize(cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (RES, RES))
    return (torch.from_numpy(im).permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0)
print(f"[flux-cn] {len(pairs)} pairs @ {RES}px, {MAX_STEPS} steps, accum {ACCUM}", flush=True)

sf, sc = vae.config.shift_factor, vae.config.scaling_factor
def encode(img):
    with torch.no_grad():
        l = (vae.encode(img.to(DEV, DT)).latent_dist.sample() - sf) * sc
    B, C, H, W = l.shape
    return _pack(l, B, C, H, W), (C, H, W)

img_ids = None; step = 0; t0 = time.time(); opt.zero_grad()
while step < MAX_STEPS:
    random.shuffle(pairs)
    for pr in pairs:
        pix, (C, H, W) = encode(load_img(pr["target_path"]))
        ctrl, _ = encode(load_img(pr["source_path"]))
        if img_ids is None: img_ids = _img_ids(H, W)
        noise = torch.randn_like(pix)
        t = torch.sigmoid(torch.randn((1,), device=DEV, dtype=DT))
        noisy = (1 - t.view(-1, 1, 1)) * pix + t.view(-1, 1, 1) * noise
        target = noise - pix
        g = torch.full((1,), 3.5, device=DEV, dtype=DT)
        cbs, csbs = controlnet(hidden_states=noisy, controlnet_cond=ctrl, timestep=t, guidance=g,
                               pooled_projections=ppe, encoder_hidden_states=pe, txt_ids=tids, img_ids=img_ids, return_dict=False)
        pred = transformer(hidden_states=noisy, timestep=t, guidance=g, pooled_projections=ppe, encoder_hidden_states=pe,
                           controlnet_block_samples=cbs, controlnet_single_block_samples=csbs,
                           txt_ids=tids, img_ids=img_ids, return_dict=False)[0]
        loss = F.mse_loss(pred.float(), target.float())
        (loss / ACCUM).backward()
        ema = loss.item() if step == 0 else 0.97 * ema + 0.03 * loss.item()
        if (step + 1) % ACCUM == 0:
            opt.step(); opt.zero_grad()
        if step % 50 == 0:
            print(f"  step {step} loss {loss.item():.4f} ema {ema:.4f} | {(time.time()-t0)/max(step,1):.1f}s/step | VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
        if step and step % SAVE_EVERY == 0:
            torch.save(controlnet.state_dict(), f"{OUT}/controlnet_step{step}.pth")
        step += 1
        if step >= MAX_STEPS: break
torch.save(controlnet.state_dict(), f"{OUT}/controlnet_final.pth")
print(f"[flux-cn] done {step} steps in {(time.time()-t0)/60:.1f}min -> {OUT}", flush=True)
