"""De-risk gate for BBDM: does the vq-f4 autoencoder (that LBBDM-f4 would use, frozen) even
reconstruct bacterial-colony texture? Round-trips multiplexed real images through it, reports
LPIPS(alex)+MSE, and saves an orig-vs-recon montage. If recon LPIPS is well below the ~0.14
realism gap we are trying to close, the VQGAN is NOT the bottleneck (mirrors the earlier SD-VAE
diagnostic where the 4ch VAE reconstructed multiplexed at ~0.02 LPIPS). Run in the `bbdm` env."""
import os, sys, glob, argparse
sys.path.insert(0, "/hpc/group/youlab/sa603/code/BBDM")
import numpy as np, cv2, torch, lpips
from model.VQGAN.vqgan import VQModel

OUT = os.environ.get("OUT", "/hpc/group/youlab/sa603/code/Data_augmentation/backbone_derisk_out")
os.makedirs(OUT, exist_ok=True)
CKPT = "/hpc/group/youlab/sa603/code/BBDM/results/VQGAN/model.ckpt"
REALS = "/hpc/group/youlab/sa603/data/multiplexed_eval/reals"
N = 12

def ns(**kw):
    n = argparse.Namespace()
    for k, v in kw.items():
        setattr(n, k, v)
    return n

# vq-f4 architecture, taken verbatim from BBDM configs/Template-LBBDM-f4.yaml (model.VQGAN.params)
ddconfig = ns(double_z=False, z_channels=3, resolution=256, in_channels=3, out_ch=3,
              ch=128, ch_mult=(1, 2, 4), num_res_blocks=2, attn_resolutions=[], dropout=0.0)
lossconfig = ns(target="torch.nn.Identity")
vqgan = VQModel(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=8192, embed_dim=3,
                ckpt_path=CKPT).eval().cuda()
for p in vqgan.parameters():
    p.requires_grad_(False)

files = sorted(glob.glob(f"{REALS}/*.TIF"))[:N]
xs = []
for fp in files:
    im = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (256, 256))
    xs.append(torch.from_numpy(im).float().permute(2, 0, 1) / 127.5 - 1.0)  # [-1,1]
x = torch.stack(xs).cuda()

loss_fn = lpips.LPIPS(net="alex").cuda()
with torch.no_grad():
    xrec, _ = vqgan(x)            # forward = encode -> quantize -> decode
    xrec = xrec.clamp(-1, 1)
    d = loss_fn(x, xrec).flatten().cpu().numpy()
    mse = ((x - xrec) ** 2).mean().item()
print(f"[bbdm vqgan recon] n={len(files)} LPIPS_alex mean={d.mean():.4f} std={d.std():.4f} "
      f"min={d.min():.4f} max={d.max():.4f} MSE={mse:.5f}", flush=True)
print("[context] SD 4ch-VAE multiplexed recon was ~0.02 LPIPS; realism gap to close ~0.14. "
      "VQGAN is fine if recon LPIPS is well below 0.14.", flush=True)

def to_u8(t):
    return ((t.clamp(-1, 1).cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)

k = min(8, len(files))
rows = [np.concatenate([cv2.cvtColor(to_u8(t[i]), cv2.COLOR_RGB2BGR) for i in range(k)], axis=1)
        for t in (x[:k], xrec[:k])]
cv2.imwrite(f"{OUT}/bbdm_vqgan_recon_montage.png", np.concatenate(rows, axis=0))
print(f"[bbdm vqgan recon] montage (top=real, bottom=recon) -> {OUT}/bbdm_vqgan_recon_montage.png", flush=True)
