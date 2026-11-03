"""Sample-image montages for the Kinshuk report: for each dataset, rows = example real sources,
columns = Real | Rotated (classic aug) | AI synthetic x3. Run on DCC (has clean data + cv2/matplotlib)."""
import os, glob
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/hpc/group/youlab/sa603/data/clean"
OUT = "reporting/p5_downstream_report/figs"; os.makedirs(OUT, exist_ok=True)
def load(fp): return cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
def rot(im, a=47):
    h, w = im.shape[:2]; M = cv2.getRotationMatrix2D((w/2, h/2), a, 1.0)
    return cv2.warpAffine(im, M, (w, h))

def pick(real_glob, to_pref, synth_dir, n=3):
    out = []
    for rfp in sorted(glob.glob(real_glob)):
        stem = os.path.splitext(os.path.basename(rfp))[0]
        s = sorted(glob.glob(f"{synth_dir}/{to_pref(stem)}_s*.png"))
        if s:
            out.append((rfp, s))
        if len(out) >= n:
            break
    return out

def montage(name, triples, title):
    n = len(triples)
    if n == 0:
        print("SKIP (no samples):", name); return
    cols = ["Real image", "Rotated\n(classic aug)", "AI synthetic #1", "AI synthetic #2", "AI synthetic #3"]
    fig, axes = plt.subplots(n, 5, figsize=(11, 2.3*n))
    if n == 1: axes = [axes]
    for i, (rfp, sfps) in enumerate(triples):
        r = load(rfp)
        imgs = [r, rot(r)] + [load(s) for s in sfps[:3]]
        for j in range(5):
            ax = axes[i][j]; ax.axis("off")
            if j < len(imgs): ax.imshow(imgs[j])
            if i == 0: ax.set_title(cols[j], fontsize=10)
    fig.suptitle(title, fontsize=13, y=0.99)
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=130, bbox_inches="tight"); plt.close()
    print("WROTE", name, "rows=", n)

montage("samples_branching",
        pick(f"{D}/branching_256/canon/*.TIF", lambda s: s, f"{D}/branching_downstream/synth_train_clean"),
        "BRANCHING colonies  -  real vs rotation vs AI-synthetic")
montage("samples_mux",
        pick(f"{D}/multiplexed_eval/reals/*.TIF", lambda s: s.replace("_Rep", "_src"), f"{D}/multiplexed_eval/synth_clean"),
        "MUX patterns (aTc=red petals, IPTG=green fill)  -  real vs rotation vs AI-synthetic")
montage("samples_kl",
        pick(f"{D}/ood_kl_2species/*.TIF", lambda s: s, f"{D}/ood_kl_2species/synth_clean"),
        "KL two-species mixing  -  real vs rotation vs AI-synthetic")
montage("samples_rattray",
        pick(f"{D}/external_datasets/rattray_2023/processed/*.png", lambda s: s, f"{D}/external_datasets/rattray_2023/synth_clean"),
        "RATTRAY strains  -  real vs rotation vs AI-synthetic")
print("done")
