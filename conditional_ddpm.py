"""Conditional DDPM (Palette-style channel-concat): source colony -> plausible replicate. The
unconditional DDPM beat SD1.5 on distribution match (CMMD 2.09 vs 3.38); conditioning it is the
ceiling-breaker for realism. UNet sees [noisy_target(3ch) | clean_source(3ch)] = 6ch -> predicts
3ch noise. Cosine schedule, DDIM sampling (eta = free diversity knob).
Env: MODE[train|gen], DATA_JSON, STEPS, OUT, CKPT, SOURCES, NUM[4], STEPS_SAMPLE[50], ETA[0.0]."""
import os, json, math, glob, copy, random
import torch, torch.nn.functional as F, numpy as np
from PIL import Image
from denoising_diffusion_pytorch import Unet

DEV = "cuda"; RES = 256; T = 1000
def cosine_beta(T, s=0.008):
    x = torch.linspace(0, T, T + 1); ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2; ac = ac / ac[0]
    return (1 - ac[1:] / ac[:-1]).clamp(1e-8, 0.999)
betas = cosine_beta(T).to(DEV); alphas = 1 - betas; acp = torch.cumprod(alphas, 0)
sa = torch.sqrt(acp); sma = torch.sqrt(1 - acp)

unet = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=6, out_dim=3, flash_attn=False).to(DEV)

def load_img(p):
    im = np.array(Image.open(p).convert("RGB").resize((RES, RES)))  # HWC RGB uint8
    return torch.from_numpy(im).permute(2, 0, 1).float() / 127.5 - 1

MODE = os.environ.get("MODE", "train")

if MODE == "train":
    pairs = [json.loads(l) for l in open(os.environ["DATA_JSON"]) if l.strip()]
    STEPS = int(os.environ.get("STEPS", "40000")); OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
    opt = torch.optim.Adam(unet.parameters(), lr=8e-5)
    ema = copy.deepcopy(unet); [p.requires_grad_(False) for p in ema.parameters()]
    step = 0
    latest = sorted(glob.glob(f"{OUT}/unet_*.pth"), key=lambda x: int(x.split("_")[-1][:-4]))
    if latest:
        unet.load_state_dict(torch.load(latest[-1], map_location=DEV)); ema.load_state_dict(torch.load(latest[-1].replace("unet_", "ema_"), map_location=DEV))
        step = int(latest[-1].split("_")[-1][:-4]); print(f"[cond-ddpm] resume step {step}", flush=True)
    BS = 8; unet.train()
    print(f"[cond-ddpm] {len(pairs)} pairs, train to {STEPS} (6ch conditional)", flush=True)
    while step < STEPS:
        random.shuffle(pairs)
        for i in range(0, len(pairs) - BS + 1, BS):
            b = pairs[i:i + BS]
            tgt = torch.stack([load_img(x["target_path"]) for x in b]).to(DEV)
            src = torch.stack([load_img(x["source_path"]) for x in b]).to(DEV)
            t = torch.randint(0, T, (tgt.shape[0],), device=DEV)
            noise = torch.randn_like(tgt)
            noisy = sa[t][:, None, None, None] * tgt + sma[t][:, None, None, None] * noise
            pred = unet(torch.cat([noisy, src], 1), t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pe, pp in zip(ema.parameters(), unet.parameters()): pe.mul_(0.995).add_(pp, alpha=0.005)
            if step % 500 == 0: print(f"  step {step} loss {loss.item():.4f}", flush=True)
            if step and step % 10000 == 0:
                torch.save(unet.state_dict(), f"{OUT}/unet_{step}.pth"); torch.save(ema.state_dict(), f"{OUT}/ema_{step}.pth")
            step += 1
            if step >= STEPS: break
    torch.save(ema.state_dict(), f"{OUT}/ema_final.pth"); print("DONE train", flush=True)

elif MODE == "gen":
    unet.load_state_dict(torch.load(os.environ["CKPT"], map_location=DEV)); unet.eval()
    SRC = os.environ.get("SOURCES", "/hpc/group/youlab/sa603/data/branching_256/test/sources")
    OUT = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)
    NUM = int(os.environ.get("NUM", "4")); S = int(os.environ.get("STEPS_SAMPLE", "50")); ETA = float(os.environ.get("ETA", "0.0"))
    ts = torch.linspace(T - 1, 0, S).long().to(DEV)
    @torch.no_grad()
    def ddim(src):
        x = torch.randn(src.shape[0], 3, RES, RES, device=DEV)
        for j in range(S):
            i = ts[j]; pred = unet(torch.cat([x, src], 1), i.repeat(src.shape[0]))
            ac = acp[i]; x0 = ((x - torch.sqrt(1 - ac) * pred) / torch.sqrt(ac)).clamp(-1, 1)
            if j < S - 1:
                ac2 = acp[ts[j + 1]]
                sig = ETA * torch.sqrt((1 - ac2) / (1 - ac) * (1 - ac / ac2))
                x = torch.sqrt(ac2) * x0 + torch.sqrt((1 - ac2 - sig ** 2).clamp(min=0)) * pred + sig * torch.randn_like(x)
            else: x = x0
        return x
    for fp in sorted(glob.glob(f"{SRC}/*.TIF")):
        cond = os.path.basename(fp).split("_")[0]
        src = load_img(fp)[None].to(DEV).repeat(NUM, 1, 1, 1)
        imgs = ddim(src)
        for k in range(NUM):
            im = ((imgs[k].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            Image.fromarray(im).save(f"{OUT}/{cond}_{k+1}.png")
    print("DONE gen ->", OUT, flush=True)
