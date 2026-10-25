"""P3 realism descriptor: CMMD(synth vs clean reals) for all 8 clean configs -> TSV.

SECONDARY to the faithfulness probe. Per this program's own finding, CMMD does NOT predict downstream
utility, so this is a realism SANITY (and a base-vs-dropout realism delta), not the pruning criterion.
Reuses pipeline/metrics/cmmd.py (paper CLIP-L/14@336). Caps synth at CAP (fixed sample) for speed;
within a dataset, base vs dropout see the SAME reals + same cap -> the delta is comparable.

Env: PYTHONPATH must include pipeline/metrics (for `import cmmd`); OUT_TSV, CAP[1000]. Runs on DCC GPU.
"""
import os, glob, csv, random
import numpy as np, torch, cv2
import cmmd

D = "/hpc/group/youlab/sa603/data/clean"
CAP = int(os.environ.get("CAP", "1000"))
OUT = os.environ.get("OUT_TSV", "faithfulness_out/cmmd_clean.tsv")
dev = "cuda" if torch.cuda.is_available() else "cpu"

# task -> (reals glob incl ext, synth dir template {v}=clean|clean_dropout)
CFG = {
    "branching": (f"{D}/branching_256/test/reals/*.TIF",             f"{D}/branching_downstream/synth_train_{{v}}"),
    "mux":       (f"{D}/multiplexed_eval/reals/*.TIF",               f"{D}/multiplexed_eval/synth_{{v}}"),
    "kl":        (f"{D}/ood_kl_2species/*.TIF",                      f"{D}/ood_kl_2species/synth_{{v}}"),
    "rattray":   (f"{D}/external_datasets/rattray_2023/processed/*.png", f"{D}/external_datasets/rattray_2023/synth_{{v}}"),
}


def load(files):
    return [cv2.cvtColor(cv2.imread(f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for f in files]


rows = []
for task, (rglob, stmpl) in CFG.items():
    reals = load(sorted(glob.glob(rglob)))
    er = cmmd.clip_embed_rgb(reals, dev)
    for v in ("clean", "clean_dropout"):
        sfiles = sorted(glob.glob(stmpl.format(v=v) + "/*.png"))
        capped = len(sfiles) > CAP
        if capped:
            random.seed(0); sfiles = sorted(random.sample(sfiles, CAP))
        eg = cmmd.clip_embed_rgb(load(sfiles), dev)
        val = cmmd.cmmd_from_embeds(er, eg)
        print(f">>> {task:<10} {v:<14} CMMD={val:>8.3f}  (n_real={len(reals)} n_synth={len(sfiles)}{' capped' if capped else ''})", flush=True)
        rows.append((task, v, len(reals), len(sfiles), f"{val:.4f}"))

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["task", "variant", "n_real", "n_synth", "cmmd"]); w.writerows(rows)
print(f"\n-> {OUT}  (CAP={CAP}; LOWER cmmd = closer to reals)")
