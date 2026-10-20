"""Fine-tune PixCell-256's Cell-ControlNet on our multiplexed replicate pairs (source -> target).

Warm-starts from the released Cell-ControlNet (it already maps a spatial image -> microscopy output,
far more data-efficient on our ~1932 pairs than training a ControlNet from base). Freezes the VAE and
the base denoiser (transformer); trains ONLY the ControlNet. Conditioning wiring mirrors the pipeline:
  controlnet(hidden_states=noisy_target_latent, conditioning=source_latent, encoder_hidden_states=NULL_UNI, timestep)
  transformer(noisy, encoder_hidden_states=NULL_UNI, controlnet_outputs=..., timestep) -> eps_pred[:, :16]
UNI = learned NULL token (sidesteps the pathology encoder). DDPM epsilon-MSE loss. Saves controlnet weights.
Run: python pixcell_train_cn.py   (pixcell env; PYTHONPATH includes .../PixCell/controlnet)
Env: EPOCHS[40] ACCUM[8] LR[1e-5] MAX_STEPS[0=full;>0 smoke] SAVE_EVERY[10] OUT[...]."""
import os, sys, json
sys.path.insert(0, "/hpc/group/youlab/sa603/code/PixCell/controlnet")
import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader
from diffusers import DiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm

TRAIN_JSON = os.environ.get("TRAIN_JSON", "/hpc/group/youlab/sa603/data/multiplexed_eval/train_multiplexed.json")
OUT = os.environ.get("OUT", "/hpc/group/youlab/sa603/code/Data_augmentation/pixcell_cn_runs/mplex")
EPOCHS = int(os.environ.get("EPOCHS", "40"))
ACCUM = int(os.environ.get("ACCUM", "8"))
LR = float(os.environ.get("LR", "1e-5"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))     # >0 => smoke
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "10"))  # epochs
os.makedirs(OUT, exist_ok=True)

def load256(fp):
    im = cv2.imread(fp, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"read failed: {fp}")
    if im.shape[:2] != (256, 256):
        im = cv2.resize(im, (256, 256))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(im).float().permute(2, 0, 1) / 255.0  # [0,1] CHW

class Pairs(Dataset):
    def __init__(self, js):
        self.pairs = [json.loads(l) for l in open(js)]
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        p = self.pairs[i]
        return load256(p["target_path"]), load256(p["source_path"])   # (target, source)

acc = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=ACCUM)

pipe = DiffusionPipeline.from_pretrained(
    "StonyBrook-CVLab/PixCell-256-Cell-ControlNet",
    custom_pipeline="StonyBrook-CVLab/PixCell-pipeline-ControlNet",
    trust_remote_code=True)  # fp32 weights; accelerate handles fp16 autocast
vae, transformer, controlnet, scheduler = pipe.vae, pipe.transformer, pipe.controlnet, pipe.scheduler
vae_scale = vae.config.scaling_factor
vae_shift = getattr(vae.config, "shift_factor", 0)

vae.requires_grad_(False)
transformer.requires_grad_(False)
controlnet.train()
if hasattr(controlnet, "transformer") and hasattr(controlnet.transformer, "pos_embed"):
    controlnet.transformer.pos_embed.requires_grad_(False)  # input patch embed frozen (mirrors upstream)
n_train = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
acc.print(f"[pixcell-cn] trainable controlnet params: {n_train/1e6:.1f}M")

null_uni = pipe.get_unconditional_embedding(1)   # (1,1,1536)

ds = Pairs(TRAIN_JSON)
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4)
opt = torch.optim.AdamW(controlnet.parameters(), lr=LR)
total = MAX_STEPS if MAX_STEPS > 0 else len(dl) * EPOCHS
sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=min(500, max(1, total // 10)), num_training_steps=total)

transformer, controlnet, vae, opt, dl, sched = acc.prepare(transformer, controlnet, vae, opt, dl, sched)
dev = acc.device
null_uni = null_uni.to(dev)
alphas_cumprod = scheduler.alphas_cumprod.to(dev)
acc.print(f"[pixcell-cn] {len(ds)} pairs | epochs={EPOCHS} accum={ACCUM} lr={LR} total_steps={total} max_steps={MAX_STEPS}")

gstep, done = 0, False
for epoch in range(EPOCHS):
    pbar = tqdm(total=len(dl), disable=not acc.is_local_main_process, desc=f"ep{epoch}")
    for target_img, source_img in dl:
        with torch.no_grad():
            tgt_lat = (vae.encode((2 * target_img - 1).to(dev)).latent_dist.mean - vae_shift) * vae_scale
            src_lat = (vae.encode((2 * source_img - 1).to(dev)).latent_dist.mean - vae_shift) * vae_scale
        bs = tgt_lat.shape[0]
        t = torch.randint(0, 1000, (bs,), device=dev, dtype=torch.long)
        atbar = alphas_cumprod[t].view(-1, 1, 1, 1)
        eps = torch.randn_like(tgt_lat)
        noisy = torch.sqrt(atbar) * tgt_lat + torch.sqrt(1 - atbar) * eps
        uni = null_uni.expand(bs, -1, -1)
        with acc.accumulate(controlnet):
            cn_out = controlnet(hidden_states=noisy, conditioning=src_lat,
                                encoder_hidden_states=uni, timestep=t, return_dict=False)[0]
            eps_pred = transformer(noisy, encoder_hidden_states=uni, controlnet_outputs=cn_out,
                                   timestep=t, added_cond_kwargs={}, return_dict=False)[0][:, :16]
            loss = ((eps_pred - eps) ** 2).mean()
            acc.backward(loss)
            if acc.sync_gradients:
                acc.clip_grad_norm_(controlnet.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        gstep += 1
        pbar.update(1); pbar.set_postfix(loss=float(loss.detach()), step=gstep)
        if MAX_STEPS > 0 and gstep >= MAX_STEPS:
            done = True; break
    if acc.is_main_process and ((epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1 or done):
        pth = f"{OUT}/controlnet_ep{epoch+1}.pth"
        torch.save(acc.unwrap_model(controlnet).state_dict(), pth)
        acc.print(f"[save] {pth}")
    if done:
        break
acc.print("[pixcell-cn] training done")
