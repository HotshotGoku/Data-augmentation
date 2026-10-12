"""Direct CMMD(gen vs reals) for UNCONDITIONAL samples (eval_metrics is condition-paired, so it nulls
on de-novo GAN images). Uses cmmd.py's own CLIP-MMD -> value comparable to our trained references.
Plus a global diversity estimate (mean pairwise LPIPS over random gen pairs). Env: G (gen dir)."""
import os, glob, cv2, torch, numpy as np
import cmmd
REALS = os.environ.get("REALS", "/hpc/group/youlab/sa603/data/branching_256/test/reals")
G = os.environ.get("G", "/hpc/group/youlab/sa603/code/Data_augmentation/gan_runs/gen/lwgan_m2")
dev = "cuda" if torch.cuda.is_available() else "cpu"

def load(d, pat):
    return [cv2.cvtColor(cv2.imread(f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            for f in sorted(glob.glob(os.path.join(d, pat)))]

reals = load(REALS, "*.TIF"); gens = load(G, "*.png")
er = cmmd.clip_embed_rgb(reals, dev); eg = cmmd.clip_embed_rgb(gens, dev)
val = float(cmmd.cmmd_from_embeds(er, eg))
print(f"\n>>> CMMD(gen vs reals) = {val:.3f}   (n_real={len(reals)} n_gen={len(gens)})", flush=True)

# global diversity: mean pairwise LPIPS over random gen pairs (caveat: GLOBAL, not per-source)
try:
    import lpips, random
    lp = lpips.LPIPS(net="alex").to(dev).eval()
    def t(im):
        x = torch.from_numpy(im).float().permute(2,0,1)[None]/127.5 - 1
        return x.to(dev)
    random.seed(0); ds = []
    idx = list(range(len(gens)))
    for _ in range(min(120, len(gens))):
        a, b = random.sample(idx, 2)
        with torch.no_grad(): ds.append(float(lp(t(gens[a]), t(gens[b]))))
    print(f">>> global diversity (pairwise LPIPS) = {np.mean(ds):.3f}  (±{np.std(ds):.3f})", flush=True)
except Exception as e:
    print(">>> diversity skipped:", e)

print(">>> REFERENCE (trained): SD1.5 CMMD 3.38 | Flux-bf16 4.84 | PixCell-branching ~50 (washed)", flush=True)
