"""VAE reconstruction diagnostic (Phase 4a of the flow-matching pilot).
Hypothesis: SD1.5's 4-channel f8 VAE is the multiplexed realism ceiling. Round-trip images through
a 4ch VAE (SD1.5) vs a 16ch f8 VAE (ostris proxy / real SD3.5) and measure reconstruction fidelity.
If 16ch reconstructs sparse multiplexed dots MUCH better (while both handle branching), the VAE/backbone
is confirmed as the limiter -> full SD3.5 pilot justified.
Env flowmatch. VAE dirs via SD15_VAE, VAE16_A [ostris], VAE16_B [SD3.5]. GPU."""
import os, glob, torch, numpy as np, cv2, lpips
from diffusers import AutoencoderKL
DEV = "cuda"
DATASETS = {
    "multiplexed": "/hpc/group/youlab/sa603/data/multiplexed_eval/reals_heldout/*.TIF",
    "branching(selx)": "/hpc/group/youlab/sa603/data/generalist/test/selx/reals/*.TIF",
    "2species(kl2)": "/hpc/group/youlab/sa603/data/generalist/test/kl2/reals/*.TIF",
}
N = 50
lpf = lpips.LPIPS(net="alex").to(DEV)

def load(fp):
    im = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (256, 256)).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)

@torch.no_grad()
def rt(vae, x):
    x = x.to(DEV, vae.dtype)
    lat = vae.encode(x).latent_dist.mean
    return vae.decode(lat).sample.float().clamp(-1, 1)

def run(vae, name):
    for ds, pat in DATASETS.items():
        fps = sorted(glob.glob(pat))[:N]
        if not fps:
            continue
        ls, ms = [], []
        for fp in fps:
            x = load(fp); r = rt(vae, x)
            ls.append(lpf(x.to(DEV), r).item()); ms.append(float(((x.to(DEV) - r) ** 2).mean()))
        print(f"{name:14s} ch={vae.config.latent_channels:2d}  {ds:16s}  reconLPIPS={np.mean(ls):.4f}  MSE={np.mean(ms):.5f}  n={len(fps)}", flush=True)

for key, name in [("SD15_VAE", "SD1.5(4ch)"), ("VAE16_A", "ostris(16ch)"), ("VAE16_B", "SD3.5(16ch)")]:
    p = os.environ.get(key)
    if not p:
        continue
    try:
        vae = AutoencoderKL.from_pretrained(p, torch_dtype=torch.float32).to(DEV).eval()
        run(vae, name); del vae; torch.cuda.empty_cache()
    except Exception as e:
        print(f"{name}: LOAD FAILED ({key}={p}): {e}", flush=True)
print("=== vae diagnostic done (lower reconLPIPS = better reconstruction) ===")
