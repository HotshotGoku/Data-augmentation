"""Fine-tune a PixCell ControlNet built FROM BASE on replicate pairs. Handles PixCell-1024 and
PixCell-256 via BASE_MODEL + RES (identical recipe, only resolution differs). The ControlNet and a
ControlNet-aware denoiser are constructed from the base transformer's own config (filtered through
inspect.signature) and warm-initialized from base weights; only a LoRA on the ControlNet trains (base
DiT + VAE frozen). VAE is borrowed from the PixCell-256-Cell-ControlNet repo (the SD3.5 VAE is gated).
Null UNI token. DDPM eps loss. Grad checkpointing + fp16 (fits ~9GB at 1024, ~6GB at 256, so a single
youlab-gpu A5000 is plenty). Saves every SAVE_EVERY steps + RESUME_CKPT for resume. Run in pixcell env;
PYTHONPATH includes .../PixCell/controlnet.
Env: BASE_MODEL[PixCell-1024] RES[1024] TRAIN_JSON EPOCHS[20] MAX_STEPS[0=full;>0 smoke] SAVE_EVERY[500]
LR[1e-5] ACCUM[8] LORA[0] RESUME_CKPT[] OUT[]."""
import os, sys, json
sys.path.insert(0, "/hpc/group/youlab/sa603/code/PixCell/controlnet")
import cv2, torch
from torch.utils.data import Dataset, DataLoader
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from pixcell_controlnet import PixCellControlNet
from pixcell_controlnet_transformer import PixCellTransformer2DModelControlNet

TRAIN_JSON = os.environ.get("TRAIN_JSON", "/hpc/group/youlab/sa603/data/multiplexed_eval/train_multiplexed.json")
OUT = os.environ.get("OUT", "/hpc/group/youlab/sa603/code/Data_augmentation/pixcell1024_cn_runs/mplex")
BASE_MODEL = os.environ.get("BASE_MODEL", "StonyBrook-CVLab/PixCell-1024")  # set PixCell-256 for the same-recipe 256 control
RES_PX = int(os.environ.get("RES", "1024"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
ACCUM = int(os.environ.get("ACCUM", "8"))
LR = float(os.environ.get("LR", "1e-5"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "500"))
RESUME = os.environ.get("RESUME_CKPT", "")
LORA = os.environ.get("LORA", "0") == "1"   # LoRA => train tiny adapters (fits 24GB) instead of full 622M ControlNet
os.makedirs(OUT, exist_ok=True)

def load_img(fp):
    im = cv2.imread(fp, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"read failed: {fp}")
    im = cv2.resize(im, (RES_PX, RES_PX))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(im).float().permute(2, 0, 1) / 255.0

class Pairs(Dataset):
    def __init__(self, js): self.pairs = [json.loads(l) for l in open(js)]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        p = self.pairs[i]
        return load_img(p["target_path"]), load_img(p["source_path"])

acc = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=ACCUM)

vae = AutoencoderKL.from_pretrained("StonyBrook-CVLab/PixCell-256-Cell-ControlNet", subfolder="vae")
pipe = DiffusionPipeline.from_pretrained(BASE_MODEL, vae=vae,
    custom_pipeline="StonyBrook-CVLab/PixCell-pipeline", trust_remote_code=True,
    low_cpu_mem_usage=False, device_map=None)  # low_cpu_mem_usage/device_map needed for PixCell-1024 (y_pos_embed); harmless for 256
scheduler = pipe.scheduler
vae_scale = pipe.vae.config.scaling_factor
vae_shift = getattr(pipe.vae.config, "shift_factor", 0)

# Build ControlNet from base using the CURRENT config-based API (upstream mock is stale).
import inspect
bcfg = dict(pipe.transformer.config)
base_sd = pipe.transformer.state_dict()
def cfg_for(cls):
    ok = set(inspect.signature(cls.__init__).parameters) - {"self", "args", "kwargs"}
    return {k: v for k, v in bcfg.items() if k in ok}
cn_kwargs = cfg_for(PixCellControlNet); cn_kwargs["n_controlnet_blocks"] = 27
controlnet = PixCellControlNet(**cn_kwargs)
controlnet.load_state_dict(base_sd, strict=False)   # warm-init shared blocks/embeds from base; controlnet zero-convs stay init'd
cn_transformer = PixCellTransformer2DModelControlNet(**cfg_for(PixCellTransformer2DModelControlNet))
cn_transformer.load_state_dict(base_sd, strict=False)
pipe.transformer = cn_transformer

vae, transformer = pipe.vae, pipe.transformer
vae.requires_grad_(False); transformer.requires_grad_(False)
controlnet.train()
if hasattr(controlnet, "transformer") and hasattr(controlnet.transformer, "pos_embed"):
    controlnet.transformer.pos_embed.requires_grad_(False)
for m in (controlnet, transformer):   # grad-ckpt on raw modules, before any LoRA wrap
    if hasattr(m, "enable_gradient_checkpointing"):
        try: m.enable_gradient_checkpointing()
        except Exception as e: acc.print("grad-ckpt NA:", e)
if LORA:
    from peft import LoraConfig, get_peft_model
    controlnet = get_peft_model(controlnet, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.0))
    controlnet.print_trainable_parameters()

if RESUME and os.path.exists(RESUME):
    controlnet.load_state_dict(torch.load(RESUME, map_location="cpu"), strict=False)
    acc.print(f"[resume] {RESUME}")
acc.print(f"[pixcell1024] trainable {sum(p.numel() for p in controlnet.parameters() if p.requires_grad)/1e6:.0f}M")

try:
    null_uni = pipe.get_unconditional_embedding(1)
except Exception:
    null_uni = pipe.transformer.caption_projection.uncond_embedding.clone().reshape(1, 1, -1)

ds = Pairs(TRAIN_JSON)
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4)
opt = torch.optim.AdamW([p for p in controlnet.parameters() if p.requires_grad], lr=LR)
total = MAX_STEPS if MAX_STEPS > 0 else len(dl) * EPOCHS
sched = get_cosine_schedule_with_warmup(opt, min(500, max(1, total // 10)), total)
transformer, controlnet, vae, opt, dl, sched = acc.prepare(transformer, controlnet, vae, opt, dl, sched)
dev = acc.device
null_uni = null_uni.to(dev)
ac = scheduler.alphas_cumprod.to(dev)
acc.print(f"[pixcell1024] {len(ds)} pairs epochs={EPOCHS} accum={ACCUM} total_steps={total} res={RES_PX}")

g, done = 0, False
for ep in range(EPOCHS):
    for tgt_img, src_img in dl:
        with torch.no_grad():
            tl = (vae.encode((2 * tgt_img - 1).to(dev)).latent_dist.mean - vae_shift) * vae_scale
            sl = (vae.encode((2 * src_img - 1).to(dev)).latent_dist.mean - vae_shift) * vae_scale
        t = torch.randint(0, 1000, (tl.shape[0],), device=dev, dtype=torch.long)
        at = ac[t].view(-1, 1, 1, 1)
        eps = torch.randn_like(tl)
        noisy = torch.sqrt(at) * tl + torch.sqrt(1 - at) * eps
        uni = null_uni.expand(tl.shape[0], -1, -1)
        with acc.accumulate(controlnet):
            co = controlnet(hidden_states=noisy, conditioning=sl, encoder_hidden_states=uni, timestep=t, return_dict=False)[0]
            ep_pred = transformer(noisy, encoder_hidden_states=uni, controlnet_outputs=co, timestep=t, added_cond_kwargs={}, return_dict=False)[0][:, :16]
            loss = ((ep_pred - eps) ** 2).mean()
            acc.backward(loss)
            if acc.sync_gradients:
                acc.clip_grad_norm_(controlnet.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        g += 1
        if g % 20 == 0:
            acc.print(f"ep{ep} step{g} loss={float(loss.detach()):.4f} peakGB={torch.cuda.max_memory_allocated()/1e9:.1f}")
        if acc.is_main_process and g % SAVE_EVERY == 0:
            torch.save(acc.unwrap_model(controlnet).state_dict(), f"{OUT}/controlnet_step{g}.pth"); acc.print(f"[save] step{g}")
        if MAX_STEPS > 0 and g >= MAX_STEPS:
            done = True; break
    if acc.is_main_process and (done or ep == EPOCHS - 1):
        torch.save(acc.unwrap_model(controlnet).state_dict(), f"{OUT}/controlnet_final.pth"); acc.print("[save] final")
    if done:
        break
acc.print(f"[pixcell1024] done peakGB={torch.cuda.max_memory_allocated()/1e9:.1f}")
