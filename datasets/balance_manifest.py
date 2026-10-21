"""Balance an augmenter pair-manifest by OVERSAMPLING each condition's pairs up to the max condition's
count (user choice 2026-10-21), and repoint image paths to the clean (dish-removed) mirror.

We TRANSFORM the existing validated manifests rather than rebuild from the image dirs: this preserves
the exact rep/pairing choices that keep the augmenter leakage-safe (no downstream test rep leaks in) --
we only (a) duplicate minority-condition pairs up to the max and (b) swap /data/ -> /data/clean/.

Handles both formats: train_*.json (source_path/target_path, absolute) and train_*_cldm.jsonl
(source/target basenames + folder). kl is already balanced (50/cond) -> oversample is a no-op there.

Usage: python balance_manifest.py            # all datasets in REG
       python balance_manifest.py --selfcheck
"""
import os, json, argparse
from collections import defaultdict

DATA = "/hpc/group/youlab/sa603/data"
REG = {
    "kl": dict(json=f"{DATA}/ood_kl_2species/train_kl.json",
               jsonl=f"{DATA}/ood_kl_2species/train_kl_cldm.jsonl",
               cond=lambda b: b.split("_")[0]),                 # 0-1_12.TIF -> 0-1
    "mux": dict(json=f"{DATA}/multiplexed_eval/train_multiplexed.json",
                jsonl=f"{DATA}/multiplexed_eval/train_multiplexed_cldm.jsonl",
                cond=lambda b: b.split("_")[0]),                # R5C4_Rep10.TIF -> R5C4
    "rattray": dict(json=f"{DATA}/external_datasets/rattray_2023/train_rattray.json",
                    jsonl=f"{DATA}/external_datasets/rattray_2023/train_rattray_cldm.jsonl",
                    cond=lambda b: b.rsplit("-", 1)[0]),        # PIL002-1.png -> PIL002
}


def repoint(p):
    return p if "/sa603/data/clean/" in p else p.replace("/sa603/data/", "/sa603/data/clean/")  # idempotent


def oversample(rows, cond_of):
    """Duplicate each condition's rows cyclically up to the max condition's count."""
    g = defaultdict(list)
    for r in rows:
        g[cond_of(r)].append(r)
    mx = max(len(v) for v in g.values())
    out = []
    for v in g.values():
        out += [v[i % len(v)] for i in range(mx)]               # oversample to max (cyclic, deterministic)
    return out, mx, len(g)


def _out(path):
    cp = repoint(path); b, e = os.path.splitext(cp)
    return f"{b}_balanced{e}"


def process(name, cfg):
    # --- .json (absolute source_path/target_path) ---
    rows = [json.loads(l) for l in open(cfg["json"]) if l.strip()]
    bal, mx, nc = oversample(rows, lambda r: cfg["cond"](os.path.basename(r["source_path"])))
    for r in bal:
        r["source_path"] = repoint(r["source_path"]); r["target_path"] = repoint(r["target_path"])
    outj = _out(cfg["json"]); os.makedirs(os.path.dirname(outj), exist_ok=True)
    with open(outj, "w") as o:
        for r in bal:
            o.write(json.dumps(r) + "\n")
    # --- _cldm.jsonl (basenames + folder) ---
    outl = "(skipped)"
    if os.path.exists(cfg["jsonl"]):
        jrows = [json.loads(l) for l in open(cfg["jsonl"]) if l.strip()]
        jbal, _, _ = oversample(jrows, lambda r: cfg["cond"](os.path.basename(r["source"])))
        for r in jbal:
            if "folder" in r:
                r["folder"] = repoint(r["folder"])
        outl = _out(cfg["jsonl"])
        with open(outl, "w") as o:
            for r in jbal:
                o.write(json.dumps(r) + "\n")
    print(f"[{name}] {len(rows)} -> {len(bal)} pairs | {nc} conds @ max={mx}/cond -> {outj}  (+ {os.path.basename(outl)})", flush=True)


def _selfcheck():
    rows = [{"source_path": "/sa603/data/d/A_1.TIF", "target_path": "/sa603/data/d/A_2.TIF"},
            {"source_path": "/sa603/data/d/A_2.TIF", "target_path": "/sa603/data/d/A_1.TIF"},
            {"source_path": "/sa603/data/d/B_1.TIF", "target_path": "/sa603/data/d/B_2.TIF"}]  # A:2, B:1
    bal, mx, nc = oversample(rows, lambda r: os.path.basename(r["source_path"]).split("_")[0])
    assert mx == 2 and nc == 2 and len(bal) == 4, f"expected 4 pairs (2 conds x max 2), got {len(bal)}"
    counts = defaultdict(int)
    for r in bal:
        counts[os.path.basename(r["source_path"]).split("_")[0]] += 1
    assert counts["A"] == counts["B"] == 2, f"not balanced: {dict(counts)}"
    assert repoint("/hpc/group/youlab/sa603/data/x/y.TIF") == "/hpc/group/youlab/sa603/data/clean/x/y.TIF"
    print(f"selfcheck OK (A:2,B:1 -> both 2; total {len(bal)}; repoint /data/ -> /data/clean/)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--only", help="comma list of datasets (default all)")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); raise SystemExit
    names = a.only.split(",") if a.only else list(REG)
    for n in names:
        process(n, REG[n])
