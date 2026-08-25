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

    # ---- LPIPS: diversity, realism (vs sibling / source), baseline real-vs-real ----
    import lpips
    loss_fn = lpips.LPIPS(net="alex").to(device)
    buckets = {t: {"div": [], "sib": [], "src": [], "base": []} for t in tags}
    rows = []
    with torch.no_grad():
        for pref, gps in gen_by_prefix.items():
            t = tag_by_prefix.get(pref)
            if t is None:
                continue
            gts = [to_t(load_gen_rgb(p), device) for p in gps]
            reals = real_by_prefix.get(pref, [])
            rimgs = [load_real_rgb(fp) for fp in reals]
            rts = [to_t(im, device) for im in rimgs if im is not None]
            div = [loss_fn(gts[a], gts[b]).item() for a, b in itertools.combinations(range(len(gts)), 2)]
            src = [loss_fn(g, rts[0]).item() for g in gts] if rts else []
            sib = [loss_fn(g, r).item() for g in gts for r in rts[1:]]
            base = [loss_fn(rts[a], rts[b]).item()
                    for a, b in itertools.combinations(range(min(len(rts), 4)), 2)][:MAX_REAL_PAIRS]
            for k, v in (("div", div), ("sib", sib), ("src", src), ("base", base)):
                buckets[t][k] += v
            rows.append([pref, t, len(gps), len(reals),
                         *[round(np.mean(x), 4) if x else "" for x in (div, sib, src, base)]])

    def summ(bkeys):
        return {"FID": bkeys[0], "diversity": agg(bkeys[1]["div"]),
                "realism_vs_sibling": agg(bkeys[1]["sib"]), "realism_vs_source": agg(bkeys[1]["src"]),
                "baseline_real_vs_real": agg(bkeys[1]["base"])}
    allb = {"div": [], "sib": [], "src": [], "base": []}
    for t in tags:
        for k in allb:
            allb[k] += buckets[t][k]

    results = {
        "gen_dir": gen_dir, "n_generated": len(gen_paths),
        "overall": summ((fid_overall, allb)),
        "by_dataset": {t: summ((fid_by_tag[t], buckets[t])) for t in tags},
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "per_prefix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prefix", "dataset", "n_gen", "n_real", "diversity", "realism_vs_sibling",
                    "realism_vs_source", "baseline_real_vs_real"])
        w.writerows(sorted(rows))
    print(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}/metrics.json + per_prefix.csv")


if __name__ == "__main__":
    main()
