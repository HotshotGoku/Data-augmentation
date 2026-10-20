"""Sample N individual images from a trained lightweight-gan checkpoint (EMA generator) for scoring.
Mirrors the package's own evaluate() path. Env: LOAD[2], N[400], OUT."""
import os, torch, torchvision
from lightweight_gan.lightweight_gan import Trainer
REPO = "/hpc/group/youlab/sa603/code/Data_augmentation"
LOAD = int(os.environ.get("LOAD", "2")); N = int(os.environ.get("N", "400"))
OUT = os.environ.get("OUT", f"{REPO}/gan_runs/gen/lwgan_m{LOAD}"); os.makedirs(OUT, exist_ok=True)

t = Trainer(name=os.environ.get("NAME","lwgan_branching"), results_dir=f"{REPO}/gan_runs/results",
            models_dir=f"{REPO}/gan_runs/models", image_size=256, use_aim=False)
t.load_config(); t.load(LOAD)
ld = t.GAN.latent_dim
G = getattr(t.GAN, "GE", None) or t.GAN.G      # EMA generator preferred
print(f"[gan-gen] loaded model_{LOAD} latent_dim={ld} bs={t.batch_size}", flush=True)

idx = 0
while idx < N:
    b = min(t.batch_size, N - idx)
    lat = torch.randn(b, ld).cuda()
    imgs = t.generate_(G, lat)                 # [b,3,256,256] in [0,1]
    for j in range(imgs.shape[0]):
        torchvision.utils.save_image(imgs[j], f"{OUT}/gen_{idx:04d}.png"); idx += 1
print(f"GENERATED {idx} -> {OUT}", flush=True)
