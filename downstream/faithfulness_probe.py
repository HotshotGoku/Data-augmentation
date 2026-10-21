"""AUTHORITATIVE faithfulness metric: a REAL-trained probe's recovery of a synth set's ASSIGNED
downstream label. This is the cheap search objective the bake-off needs -- it scores a synth source
win>tie>lose per axis WITHOUT running the full (noisy, expensive) downstream utility bake-off.

Every downstream win/loss turns on label-faithfulness: does a synth image actually carry the label the
downstream trainer treats as truth? We measure it directly:
  1. train a probe on REAL images for the task's native label(s) (resnet, same transforms as the harness),
  2. hold out real examples -> probe CEILING (recovery on reals; guards the "probe just can't do it" case),
  3. score the SYNTH set -> recovery of each synth's ASSIGNED label (= faithfulness).

"Assigned label" per task:
  - decode (mux, later kl/rattray): the condition the synth was generated for (from its filename) --
    INTENDED-LABEL recovery. mux's IPTG-lose shows up as low col recovery even while aTc(row) recovery
    stays high -> a clean win/lose contrast inside ONE dataset, needing no regen.
  - morphology (branching): the labeler's recomputed feature value for that synth image (features CSV) --
    MANIFOLD-CONSISTENCY: does a real-trained probe agree with the value the harness will train on? A
    synth whose labeler-value doesn't match its real-world appearance injects a mislabeled example.

Readout: per axis, synth_recovery (cls->accuracy, reg->R2) and norm = synth_recovery / ceiling
(comparable across tasks). The metric is validated if it ranks known win > tie > lose.

Env: TASK {mux|branching}, SYNTH_DIR (required), SEED[0], EPOCHS[30], BACKBONE[resnet18|resnet50],
     REG_WEIGHT[3.0], NW[4], OUT_TSV, + task data paths (defaults = DCC). Runs on DCC GPU.
Usage: TASK=mux   SYNTH_DIR=.../multiplexed_eval/synth_ft python faithfulness_probe.py
       TASK=branching FEATURES_CSV=.../features_branching_dropout_s10g7.csv \
                      SYNTH_DIR=.../synth_train_dropout_s10g7 python faithfulness_probe.py
       python faithfulness_probe.py --selfcheck   # offline, numpy-only metric check
"""
import os, glob, re, json, csv, argparse
import numpy as np

SEED = int(os.environ.get("SEED", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BACKBONE = os.environ.get("BACKBONE", "resnet18")
REG_WEIGHT = float(os.environ.get("REG_WEIGHT", "3.0"))
NW = int(os.environ.get("NW", "4"))
BRANCH_FEATS = ["shape_factor", "fractal_dim", "solidity", "eccentricity", "edge_std"]


# ---- metrics (pure numpy; shared by eval + selfcheck) ----
def r2(pred, targ):
    pred, targ = np.asarray(pred, float), np.asarray(targ, float)
    ss_res = ((pred - targ) ** 2).sum()
    ss_tot = ((targ - targ.mean()) ** 2).sum() + 1e-9
    return float(1 - ss_res / ss_tot)

def acc(pred_cls, targ_cls):
    return float((np.asarray(pred_cls) == np.asarray(targ_cls)).mean())


# ---- task builders: -> (cls_axes[(name,nclass)], reg_axes[name], train, test, synth) ----
#      each item = (img_uint8_rgb, cls_labels[int...], reg_labels[float...])
def _load_rgb(fp):
    import cv2
    return cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def build_mux():
    REALS = os.environ.get("REALS_DIR", "/hpc/group/youlab/sa603/data/multiplexed_eval/reals")
    SYNTH = os.environ["SYNTH_DIR"]
    LABELS = os.environ.get("LABELS", "/hpc/group/youlab/ks723/storage/Exp_images/Multiplexed_patterning/labels_aTc_IPTG.json")
    TEST_REPS = {3, 7, 12}
    row2atc, col2iptg = {}, {}
    for line in open(LABELS):
        d = json.loads(line)
        m = re.match(r"Rep\d+_Row(\d+)_Column(\d+)\.TIF", d["source"])
        if m:
            row2atc[int(m.group(1))] = float(d["target"][0])
            col2iptg[int(m.group(2))] = float(d["target"][1])
    cls_axes, reg_axes = [("row", 10), ("col", 7)], ["atc", "iptg"]

    def item(fp, r, c):
        return (_load_rgb(fp), [r - 1, c - 1], [row2atc[r], col2iptg[c]])

    train, test = [], []
    for fp in sorted(glob.glob(REALS + "/*.TIF")):
        m = re.match(r"R(\d+)C(\d+)_Rep(\d+)\.TIF$", os.path.basename(fp))
        if not m:
            continue
        r, c, rep = int(m.group(1)), int(m.group(2)), int(m.group(3))
        (test if rep in TEST_REPS else train).append(item(fp, r, c))
    synth = []
    for fp in sorted(glob.glob(SYNTH + "/*.png")):
        m = re.match(r"R(\d+)C(\d+)_src\d+_s\d+\.png$", os.path.basename(fp))
        if m:
            synth.append(item(fp, int(m.group(1)), int(m.group(2))))
    return cls_axes, reg_axes, train, test, synth


def build_branching():
    TRAIN_REAL = os.environ.get("TRAIN_REAL_DIR", "/hpc/group/youlab/sa603/data/branching_256/canon")
    TEST_SRC = os.environ.get("TEST_SRC_DIR", "/hpc/group/youlab/sa603/data/branching_256/test/sources")
    TEST_REAL = os.environ.get("TEST_REAL_DIR", "/hpc/group/youlab/sa603/data/branching_256/test/reals")
    SYNTH = os.environ["SYNTH_DIR"]
    FEATURES_CSV = os.environ["FEATURES_CSV"]          # reals + the synth variant being scored
    labels = {row["name"]: [float(row[f]) for f in BRANCH_FEATS] for row in csv.DictReader(open(FEATURES_CSV))}
    cls_axes, reg_axes = [], BRANCH_FEATS

    # held-out test conditions (the 28 the augmenter never trained on) = probe ceiling
    test_by = {}
    for fp in glob.glob(TEST_SRC + "/*_src.TIF") + glob.glob(TEST_REAL + "/*_Rep*.TIF"):
        b = os.path.basename(fp)
        if b in labels:
            test_by.setdefault(b.split("_")[0], []).append((fp, b))
    test_conds = set(test_by)
    train = []
    for fp in glob.glob(TRAIN_REAL + "/*_Rep*.TIF") + glob.glob(TRAIN_REAL + "/*_src.TIF"):
        b = os.path.basename(fp)
        if b.split("_")[0] in test_conds or b not in labels:
            continue
        train.append((_load_rgb(fp), [], labels[b]))
    test = [(_load_rgb(fp), [], labels[b]) for c in test_by for fp, b in test_by[c]]
    synth = []
    for fp in sorted(glob.glob(SYNTH + "/*.png")):
        b = os.path.basename(fp)
        if b in labels:
            synth.append((_load_rgb(fp), [], labels[b]))
    return cls_axes, reg_axes, train, test, synth


BUILDERS = {"mux": build_mux, "branching": build_branching}


def run(task):
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    cls_axes, reg_axes, train, test, synth = BUILDERS[task]()
    if not train or not synth:
        raise SystemExit(f"[{task}] need real train + synth: got {len(train)} train, {len(synth)} synth")
    print(f"[{task}] cls_axes={cls_axes} reg_axes={reg_axes} | real train={len(train)} test={len(test)} synth={len(synth)}", flush=True)

    # standardize reg targets on real train (R2 is scale-invariant; keeps multi-feature MSE balanced)
    RTR = np.array([it[2] for it in train], float) if reg_axes else np.zeros((len(train), 0))
    mu, sd = RTR.mean(0), RTR.std(0) + 1e-6

    NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    TRAIN_TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                   transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORM])
    EVAL_TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                  transforms.ToTensor(), NORM])

    class DS(Dataset):
        def __init__(self, items, tf): self.items, self.tf = items, tf
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            img, cl, rg = self.items[i]
            return (self.tf(img), torch.tensor(cl, dtype=torch.long),
                    torch.tensor(rg, dtype=torch.float32))  # reg kept RAW; standardized in the loss

    tl = DataLoader(DS(train, TRAIN_TF), batch_size=32, shuffle=True, num_workers=NW, drop_last=len(train) > 32)
    tel = DataLoader(DS(test, EVAL_TF), batch_size=64, num_workers=NW) if test else None
    syl = DataLoader(DS(synth, EVAL_TF), batch_size=64, num_workers=NW)

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            if BACKBONE == "resnet50":
                ctor, weights, feat = resnet50, ResNet50_Weights.IMAGENET1K_V2, 2048
            else:
                ctor, weights, feat = resnet18, ResNet18_Weights.IMAGENET1K_V1, 512
            try: self.b = ctor(weights=weights)
            except Exception as e: print("[warn] no pretrained weights:", e, flush=True); self.b = ctor(weights=None)
            self.b.fc = nn.Identity()
            self.cls_heads = nn.ModuleList(nn.Linear(feat, n) for _, n in cls_axes)
            self.reg_head = nn.Linear(feat, len(reg_axes)) if reg_axes else None
        def forward(self, x):
            f = self.b(x)
            return [h(f) for h in self.cls_heads], (self.reg_head(f) if self.reg_head is not None else None)

    net = Probe().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEV)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=DEV)

    for ep in range(EPOCHS):
        net.train()
        for x, cl, rg in tl:
            x, cl, rg = x.to(DEV), cl.to(DEV), rg.to(DEV)
            cls_logits, reg_pred = net(x)
            loss = sum(ce(cls_logits[i], cl[:, i]) for i in range(len(cls_axes)))
            if reg_axes:
                loss = loss + REG_WEIGHT * mse(reg_pred, (rg - mu_t) / sd_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    @torch.no_grad()
    def score(loader):
        net.eval()
        cls_p = [[] for _ in cls_axes]; cls_t = [[] for _ in cls_axes]
        reg_p, reg_t = [], []
        for x, cl, rg in loader:
            cls_logits, reg_pred = net(x.to(DEV))
            for i in range(len(cls_axes)):
                cls_p[i].append(cls_logits[i].argmax(1).cpu().numpy()); cls_t[i].append(cl[:, i].numpy())
            if reg_axes:
                reg_p.append(reg_pred.cpu().numpy() * sd + mu); reg_t.append(rg.numpy())  # un-standardize preds
        out = {}
        for i, (name, _) in enumerate(cls_axes):
            out[name] = ("cls", acc(np.concatenate(cls_p[i]), np.concatenate(cls_t[i])))
        if reg_axes:
            P, T = np.concatenate(reg_p), np.concatenate(reg_t)
            for j, name in enumerate(reg_axes):
                out[name] = ("reg", r2(P[:, j], T[:, j]))
        return out

    ceiling = score(tel) if tel else {}
    synth_rec = score(syl)
    tag = os.path.basename(os.path.normpath(os.environ["SYNTH_DIR"]))

    print(f"\n[{task}] synth={tag} backbone={BACKBONE} seed={SEED} epochs={EPOCHS}", flush=True)
    print(f"  {'axis':<14}{'kind':>5}{'ceiling':>10}{'synth':>10}{'norm':>8}", flush=True)
    for name, (kind, sv) in synth_rec.items():
        cv = ceiling.get(name, (kind, float('nan')))[1]
        norm = sv / cv if cv not in (0, float('nan')) and cv == cv else float('nan')
        print(f"  {name:<14}{kind:>5}{cv:>10.3f}{sv:>10.3f}{norm:>8.3f}", flush=True)

    OUT = os.environ.get("OUT_TSV")
    if OUT:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            if new:
                w.writerow(["task", "synth_tag", "backbone", "seed", "n_real_train", "axis", "kind", "ceiling", "synth_recovery", "norm"])
            for name, (kind, sv) in synth_rec.items():
                cv = ceiling.get(name, (kind, float('nan')))[1]
                norm = sv / cv if cv == cv and cv != 0 else float('nan')
                w.writerow([task, tag, BACKBONE, SEED, len(train), name, kind, f"{cv:.4f}", f"{sv:.4f}", f"{norm:.4f}"])
        print(f"  -> appended {len(synth_rec)} rows to {OUT}", flush=True)


def _selfcheck():
    rng = np.random.default_rng(0)
    # reg: faithful recovery (pred tracks label) scores high; unfaithful (shuffled) scores low
    y = rng.normal(size=200)
    assert r2(y + rng.normal(0, 0.1, 200), y) > 0.9, "faithful reg should be ~1"
    assert r2(rng.permutation(y), y) < 0.3, "shuffled reg should be low"
    # cls: faithful=1.0 > shuffled
    c = rng.integers(0, 10, 200)
    assert acc(c, c) == 1.0
    assert acc(rng.permutation(c), c) < 0.3
    assert r2(y + rng.normal(0, 0.1, 200), y) > r2(rng.permutation(y), y), "metric must rank faithful>unfaithful"
    print("selfcheck OK (reg faithful>shuffled, cls faithful=1.0>shuffled; metric discriminates)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=os.environ.get("TASK"))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); raise SystemExit
    if a.task not in BUILDERS:
        raise SystemExit(f"set TASK or --task to one of {list(BUILDERS)} (got {a.task!r})")
    run(a.task)
