"""
eval_metrics.py — quantitative eval of the reptorep augmenter's generated replicates.

Metrics (all computed overall AND broken down per dataset/tag):
  1) FID                 — realism: distribution of generated vs real replicates
                           (reals preprocessed identically to the model's source).
  2) diversity_LPIPS     — LPIPS between the 2 samples the model made from one source.
                           HIGHER = more varied replicates (evidence against mode collapse).
  3) realism_vs_sibling  — LPIPS between generated samples and real *other* replicates
                           of the same condition. LOWER = looks like a plausible real replicate.
  4) realism_vs_source   — LPIPS vs the exact input replicate (ControlNet fidelity).
  5) baseline_real_vs_real — LPIPS between two REAL replicates of the same condition.
                           THE YARDSTICK: if (3) ~= (5), generated is as close to real as
                           reals are to each other. If (3) >> (5), it's too far.

Point it anywhere:
  --gen_dir    inference output folder (default: latest inference/v*_REPTOREP)
  --real_spec  "TAG:/path,TAG2:/path2"  (default: the 3 in-domain test folders)
               e.g. OOD run:  --real_spec "KL2SP:/hpc/group/youlab/sa603/data/ood_kl_2species"
  --out        results dir (default: eval_metrics_results)

Notes: FID at a few-hundred images is INDICATIVE (under-sampled covariance); the per-image
LPIPS numbers are the robust ones. First run downloads Inception+AlexNet weights (nodes have net).
"""
import os, glob, json, csv, argparse, itertools, shutil
import numpy as np
import cv2
import torch

import Data_augmentation.utils.shared_resources_config as shared_resources_config  # noqa: F401  (sets sys.path)
from cldm.preprocess import preprocess_experimental_backgroundblack
from Data_augmentation.utils.local_config import (
    EXP_FOLDER_TEST, EXP_FOLDER_TEST_V3, EMRAH_EXP_FOLDER_TEST, OUTPUT_DIR_REPTOREP,
)

# --- new metrics v2 (additive; degrade gracefully if a dep is missing) ---
try:
    import cmmd as _cmmd
    _HAS_CMMD = _cmmd.available()
except Exception as _e:  # pragma: no cover
    _HAS_CMMD = False
    print(f"[eval] CMMD unavailable ({_e}); skipping.")
try:
    from persample import calculate_ssim_batch, calculate_lpips_score_batch, calculate_orb_similarity_batch
    _HAS_PS = True
except Exception as _e:  # pragma: no cover
    _HAS_PS = False
    print(f"[eval] per-sample SSIM/LPIPS-VGG/ORB unavailable ({_e}); skipping.")

RES = 256
DEFAULT_REAL = [("FINAL", EXP_FOLDER_TEST), ("KUIZHU", EXP_FOLDER_TEST_V3), ("EMRAH", EMRAH_EXP_FOLDER_TEST)]
MAX_REAL_PAIRS = 6   # cap real-vs-real pairs per prefix (bounds compute when a condition has many replicates)


def prefix_of(path):
    return os.path.basename(path).split("_")[0]


def load_gen_rgb(path):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img if img.shape[:2] == (RES, RES) else cv2.resize(img, (RES, RES))


def load_real_rgb(path):
    img = preprocess_experimental_backgroundblack(path)   # 256x256x3 RGB, same as model source
    return None if img is None else (img if img.shape[:2] == (RES, RES) else cv2.resize(img, (RES, RES)))


def to_t(img_rgb, device):
    return (torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 127.5 - 1.0).unsqueeze(0).to(device)


def to_t01(img_rgb, device):  # [1,C,H,W] in [0,1] for Kinshuk's per-sample functions
    return (torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0).unsqueeze(0).to(device)


def agg(vals):
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "n": len(vals)} if vals else None


def fid_of(real_paths, gen_paths, workdir, device):
    from pytorch_fid import fid_score
    tr, tg = os.path.join(workdir, "_fr"), os.path.join(workdir, "_fg")
    for d in (tr, tg):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    n = 0
    for fp in real_paths:
        img = load_real_rgb(fp)
        if img is None:
            continue
        cv2.imwrite(os.path.join(tr, f"{n:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); n += 1
    for i, p in enumerate(gen_paths):
        shutil.copy(p, os.path.join(tg, f"{i:05d}.png"))
    if n < 2 or len(gen_paths) < 2:
        shutil.rmtree(tr, ignore_errors=True); shutil.rmtree(tg, ignore_errors=True)
        return None
    val = fid_score.calculate_fid_given_paths([tr, tg], batch_size=50, device=device, dims=2048)
    shutil.rmtree(tr, ignore_errors=True); shutil.rmtree(tg, ignore_errors=True)
    return round(float(val), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default=None)
    ap.add_argument("--real_spec", default=None, help='"TAG:/path,TAG2:/path2" (default: 3 in-domain folders)')
    ap.add_argument("--out", default="eval_metrics_results")
    args = ap.parse_args()

    real_spec = DEFAULT_REAL if not args.real_spec else \
        [(c.split(":", 1)[0].strip(), c.split(":", 1)[1].strip()) for c in args.real_spec.split(",")]

    gen_dir = args.gen_dir or sorted(glob.glob(os.path.join(os.path.dirname(OUTPUT_DIR_REPTOREP), "v*_REPTOREP")))[-1]
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"gen_dir: {gen_dir}\nreal_spec: {real_spec}\ndevice: {device}")

    gen_paths = sorted(glob.glob(os.path.join(gen_dir, "*.png")))
    gen_by_prefix = {}
    for p in gen_paths:
        gen_by_prefix.setdefault(prefix_of(p), []).append(p)

    real_by_prefix, tag_by_prefix = {}, {}
    for tag, folder in real_spec:
        for fp in sorted(glob.glob(os.path.join(folder, "*.TIF"))):
            pref = prefix_of(fp)
            if pref not in real_by_prefix:
                real_by_prefix[pref], tag_by_prefix[pref] = [], tag
            if tag_by_prefix[pref] == tag:
                real_by_prefix[pref].append(fp)
    for k in real_by_prefix:
        real_by_prefix[k] = sorted(real_by_prefix[k])
    tags = [t for t, _ in real_spec]
    print(f"generated: {len(gen_paths)} imgs / {len(gen_by_prefix)} prefixes ; real conditions: {len(real_by_prefix)}")

    # ---- FID: overall + per tag ----
    gp_by_tag = {t: [] for t in tags}
    for p in gen_paths:
        t = tag_by_prefix.get(prefix_of(p))
        if t:
            gp_by_tag[t].append(p)
    rp_by_tag = {t: [] for t in tags}
    for pref, paths in real_by_prefix.items():
        if pref in gen_by_prefix:
            rp_by_tag[tag_by_prefix[pref]] += paths
    all_real = [fp for pref in gen_by_prefix if pref in real_by_prefix for fp in real_by_prefix[pref]]
    fid_overall = fid_of(all_real, gen_paths, args.out, device)
    fid_by_tag = {t: fid_of(rp_by_tag[t], gp_by_tag[t], args.out, device) for t in tags}
    print(f"FID overall: {fid_overall} ; by tag: {fid_by_tag}")

    # ---- LPIPS(alex): diversity, realism (sibling/source), baseline; + nearest-neighbor per-sample; + CMMD ----
    import lpips
    loss_fn = lpips.LPIPS(net="alex").to(device)
    BKEYS = ["div", "sib", "src", "base", "nn_alex", "nn_vgg", "nn_ssim", "nn_orb", "base_nn", "auth", "copy"]
    buckets = {t: {k: [] for k in BKEYS} for t in tags}
    gen_rgb_by_tag = {t: [] for t in tags}
    real_rgb_by_tag = {t: [] for t in tags}
    cmmd_by_prefix = {}
    rows, img_rows = [], []
    with torch.no_grad():
        for pref, gps in gen_by_prefix.items():
            t = tag_by_prefix.get(pref)
            if t is None:
                continue
            gimgs = [load_gen_rgb(p) for p in gps]
            gts = [to_t(im, device) for im in gimgs]
            reals = real_by_prefix.get(pref, [])
            valid = [(fp, im) for fp, im in ((fp, load_real_rgb(fp)) for fp in reals) if im is not None]
            real_paths_v = [fp for fp, _ in valid]
            rimgs = [im for _, im in valid]
            rts = [to_t(im, device) for im in rimgs]

            div = [loss_fn(gts[a], gts[b]).item() for a, b in itertools.combinations(range(len(gts)), 2)]
            src = [loss_fn(g, rts[0]).item() for g in gts] if rts else []
            sib = [loss_fn(g, r).item() for g in gts for r in rts[1:]]
            base = [loss_fn(rts[a], rts[b]).item()
                    for a, b in itertools.combinations(range(min(len(rts), 4)), 2)][:MAX_REAL_PAIRS]
            for k, v in (("div", div), ("sib", sib), ("src", src), ("base", base)):
                buckets[t][k] += v

            # pool RGB for CMMD (distribution metric); per-prefix CMMD is diagnostic (small n)
            gen_rgb_by_tag[t] += gimgs
            real_rgb_by_tag[t] += rimgs
            if _HAS_CMMD:
                cmmd_by_prefix[pref] = _cmmd.cmmd_of(rimgs, gimgs, device)

            # per-sample: match each gen to its NEAREST real sibling (alex LPIPS), then Kinshuk's SSIM/LPIPS-VGG/ORB
            nn_alex, nn_vgg, nn_ssim, nn_orb, base_nn, auth, copies = [], [], [], [], [], [], []
            if rts:
                D = np.array([[loss_fn(g, r).item() for r in rts] for g in gts])  # [G, R]
                nn_idx = D.argmin(axis=1)
                nn_alex = D.min(axis=1).tolist()
                if len(rts) >= 2:  # nearest real-vs-real floor: each real -> nearest OTHER real
                    DR = np.array([[loss_fn(rts[a], rts[b]).item() if a != b else np.inf
                                    for b in range(len(rts))] for a in range(len(rts))])
                    base_nn = DR.min(axis=1).tolist()
                if _HAS_PS:
                    gb = torch.cat([to_t01(im, device) for im in gimgs], 0)
                    nb = torch.cat([to_t01(rimgs[int(i)], device) for i in nn_idx], 0)
                    nn_ssim = calculate_ssim_batch(gb, nb).tolist()
                    nn_vgg = calculate_lpips_score_batch(gb, nb).tolist()
                    nn_orb = calculate_orb_similarity_batch(gb, nb).tolist()
                fmean = float(np.mean(base_nn)) if base_nn else None
                fmin = float(np.min(base_nn)) if base_nn else None
                auth = [a - fmean for a in nn_alex] if fmean is not None else []
                copies = [1.0 if (fmin is not None and a < fmin) else 0.0 for a in nn_alex]
                for k, v in (("nn_alex", nn_alex), ("nn_vgg", nn_vgg), ("nn_ssim", nn_ssim),
                             ("nn_orb", nn_orb), ("base_nn", base_nn), ("auth", auth), ("copy", copies)):
                    buckets[t][k] += v
                for j, gp in enumerate(gps):
                    img_rows.append([gp, pref, t, os.path.basename(real_paths_v[int(nn_idx[j])]),
                                     round(nn_alex[j], 4),
                                     round(nn_vgg[j], 4) if nn_vgg else "",
                                     round(nn_ssim[j], 4) if nn_ssim else "",
                                     round(nn_orb[j], 4) if nn_orb else "",
                                     round(auth[j], 4) if auth else "",
                                     int(copies[j]) if copies else ""])
            rows.append([pref, t, len(gps), len(reals),
                         *[round(np.mean(x), 4) if x else "" for x in (div, sib, src, base)],
                         cmmd_by_prefix.get(pref, ""),
                         *[round(np.mean(x), 4) if len(x) else "" for x in (nn_vgg, nn_ssim, nn_orb)]])

    # ---- CMMD: pooled overall + per tag (many more samples than per-prefix -> the reliable number) ----
    cmmd_overall = None
    cmmd_by_tag = {t: None for t in tags}
    if _HAS_CMMD:
        allg = [im for t in tags for im in gen_rgb_by_tag[t]]
        allr = [im for t in tags for im in real_rgb_by_tag[t]]
        cmmd_overall = _cmmd.cmmd_of(allr, allg, device)
        cmmd_by_tag = {t: _cmmd.cmmd_of(real_rgb_by_tag[t], gen_rgb_by_tag[t], device) for t in tags}
    print(f"CMMD overall: {cmmd_overall} ; by tag: {cmmd_by_tag}")

    def summ(fid, cmmd, b):
        return {"FID": fid, "CMMD": cmmd,
                "diversity": agg(b["div"]),
                "realism_vs_sibling": agg(b["sib"]), "realism_vs_source": agg(b["src"]),
                "baseline_real_vs_real": agg(b["base"]),
                "realism_nn_lpips_alex": agg(b["nn_alex"]), "realism_nn_lpips_vgg": agg(b["nn_vgg"]),
                "realism_nn_ssim": agg(b["nn_ssim"]), "realism_nn_orb": agg(b["nn_orb"]),
                "baseline_nn_real_vs_real": agg(b["base_nn"]), "authenticity_gap": agg(b["auth"]),
                "copy_rate": (round(float(np.mean(b["copy"])), 4) if b["copy"] else None)}

    allb = {k: [] for k in BKEYS}
    for t in tags:
        for k in BKEYS:
            allb[k] += buckets[t][k]

    results = {
        "gen_dir": gen_dir, "n_generated": len(gen_paths), "metrics_version": 2,
        "overall": summ(fid_overall, cmmd_overall, allb),
        "by_dataset": {t: summ(fid_by_tag[t], cmmd_by_tag[t], buckets[t]) for t in tags},
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "per_prefix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prefix", "dataset", "n_gen", "n_real", "diversity", "realism_vs_sibling",
                    "realism_vs_source", "baseline_real_vs_real",
                    "cmmd", "realism_nn_lpips_vgg", "realism_nn_ssim", "realism_nn_orb"])
        w.writerows(sorted(rows))
    with open(os.path.join(args.out, "per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen_path", "prefix", "dataset", "nearest_real", "nn_lpips_alex", "nn_lpips_vgg",
                    "nn_ssim", "nn_orb", "auth_gap", "is_possible_copy"])
        w.writerows(sorted(img_rows))
    print(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}/metrics.json + per_prefix.csv + per_image.csv")


if __name__ == "__main__":
    main()
