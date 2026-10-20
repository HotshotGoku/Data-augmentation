"""Downstream UTILITY test: read the multiplexed inputs (aTc, IPTG) out of a colony pattern.
Two framings, reported together (per Kinshuk):
  - classification: Row (10-way, = aTc level) + Column (7-way, = IPTG level)
  - regression:     predict the actual [aTc, IPTG] concentrations (Kinshuk's ground-truth labels)
Data-scarce regime; compares training-set composition:
  MODE=real       -> K real reps/condition
  MODE=classical  -> K real + N_PER rotated/flipped copies per source (matched count)
  MODE=augmenter  -> K real + N_PER fine-tuned-augmenter synth per source (matched count)
All arms share identical training transforms. Eval on held-out REAL test reps {3,7,12}; model
selection on val reps {2,6,11} PER FRAMING (classification metrics from the best-val-joint epoch,
regression metrics from the best-val-R2 epoch). Appends one row to RESULTS_TSV.

Env: K (1|2|3|all), MODE, SEED, N_PER [8], EPOCHS [40], BACKBONE [resnet18|resnet50],
     REG_WEIGHT [3.0], RESULTS_TSV, SYNTH_DIR, LABELS. DCC GPU."""
import os, glob, re, json, random, csv
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

REALS = "/hpc/group/youlab/sa603/data/multiplexed_eval/reals"
SYNTH = os.environ.get("SYNTH_DIR", "/hpc/group/youlab/sa603/data/multiplexed_eval/synth_ft")
RESULTS = os.environ.get("RESULTS_TSV", "/hpc/group/youlab/sa603/code/Data_augmentation/downstream_multiplexed_out/results.tsv")
LABELS = os.environ.get("LABELS", "/hpc/group/youlab/ks723/storage/Exp_images/Multiplexed_patterning/labels_aTc_IPTG.json")
K = os.environ.get("K", "1")
MODE = os.environ.get("MODE", "real")
MODE_TAG = os.environ.get("MODE_TAG", MODE)   # label written to results (lets us distinguish synth sources)
SEED = int(os.environ.get("SEED", "0"))
N_PER = int(os.environ.get("N_PER", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BACKBONE = os.environ.get("BACKBONE", "resnet18")
REG_WEIGHT = float(os.environ.get("REG_WEIGHT", "3.0"))
DEV = "cuda"
SRC_REPS = [8, 9, 10]                # present in all 70 conditions -> balanced K in {1,2,3}
ALL_TRAIN = [1, 4, 5, 8, 9, 10]
VAL_REPS, TEST_REPS = [2, 6, 11], [3, 7, 12]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# Kinshuk's ground truth: Row -> aTc concentration, Column -> IPTG concentration
row2atc, col2iptg = {}, {}
for line in open(LABELS):
    d = json.loads(line)
    m = re.match(r"Rep\d+_Row(\d+)_Column(\d+)\.TIF", d["source"])
    if m:
        row2atc[int(m.group(1))] = float(d["target"][0])
        col2iptg[int(m.group(2))] = float(d["target"][1])
print(f"[labels] {len(row2atc)} aTc levels, {len(col2iptg)} IPTG levels", flush=True)

def load_rgb(fp):
    return cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def rot_flip(img, ang, flip):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return cv2.flip(out, 1) if flip else out

real_by = {}
for fp in glob.glob(REALS + "/*.TIF"):
    m = re.match(r"R(\d+)C(\d+)_Rep(\d+)\.TIF$", os.path.basename(fp))
    if m:
        real_by[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = fp

use_reps = set(ALL_TRAIN) if K == "all" else set(SRC_REPS[:int(K)])

def item(img, row, col):  # (img, row0, col0, atc, iptg)
    return (img, row - 1, col - 1, row2atc[row], col2iptg[col])

train_items = []
for (row, col, rep), fp in real_by.items():
    if rep not in use_reps:
        continue
    img = load_rgb(fp)
    train_items.append(item(img, row, col))
    if MODE == "classical":
        rng = random.Random((row * 131 + col * 17 + rep * 7 + SEED) & 0xffffffff)
        for _ in range(N_PER):
            train_items.append(item(rot_flip(img, rng.uniform(0, 360), rng.random() < 0.5), row, col))
    elif MODE == "augmenter":
        for s in sorted(glob.glob(f"{SYNTH}/R{row}C{col}_src{rep}_s*.png"))[:N_PER]:
            train_items.append(item(load_rgb(s), row, col))

def eval_items(reps):
    return [item(load_rgb(fp), row, col) for (row, col, rep), fp in real_by.items() if rep in reps]
val_items, test_items = eval_items(VAL_REPS), eval_items(TEST_REPS)

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORM])
EVAL_TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                              transforms.ToTensor(), NORM])

class DS(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        img, r, c, a, g = self.items[i]
        return self.tf(img), r, c, torch.tensor([a, g], dtype=torch.float32)

tl = DataLoader(DS(train_items, TRAIN_TF), batch_size=32, shuffle=True, num_workers=4)
vl = DataLoader(DS(val_items, EVAL_TF), batch_size=64, num_workers=4)
tel = DataLoader(DS(test_items, EVAL_TF), batch_size=64, num_workers=4)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        if BACKBONE == "resnet50":
            ctor, weights, feat = resnet50, ResNet50_Weights.IMAGENET1K_V2, 2048
        else:
            ctor, weights, feat = resnet18, ResNet18_Weights.IMAGENET1K_V1, 512
        try:
            self.b = ctor(weights=weights)
        except Exception as e:
            print("[warn] pretrained weights unavailable, random init:", e, flush=True)
            self.b = ctor(weights=None)
        self.b.fc = nn.Identity()
        self.row, self.col, self.reg = nn.Linear(feat, 10), nn.Linear(feat, 7), nn.Linear(feat, 2)
    def forward(self, x):
        f = self.b(x); return self.row(f), self.col(f), self.reg(f)

net = Decoder().to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

@torch.no_grad()
def evaluate(loader):
    net.eval(); rc = cc = jc = n = 0; preds, tgts = [], []
    for x, r, c, y in loader:
        x, r, c = x.to(DEV), r.to(DEV), c.to(DEV)
        pr, pc, pg = net(x)
        pr, pc = pr.argmax(1), pc.argmax(1)
        rc += (pr == r).sum().item(); cc += (pc == c).sum().item()
        jc += ((pr == r) & (pc == c)).sum().item(); n += len(r)
        preds.append(pg.cpu().numpy()); tgts.append(y.numpy())
    P, T = np.concatenate(preds), np.concatenate(tgts)
    mae = np.abs(P - T).mean(0)                      # [aTc_mae, IPTG_mae]
    ss_res = ((P - T) ** 2).sum(0); ss_tot = ((T - T.mean(0)) ** 2).sum(0) + 1e-9
    r2 = 1 - ss_res / ss_tot                          # [aTc_r2, IPTG_r2]
    return {"row": rc / n, "col": cc / n, "joint": jc / n,
            "atc_mae": mae[0], "iptg_mae": mae[1], "atc_r2": r2[0], "iptg_r2": r2[1]}

# Per-framing model selection: classification snapshot at best val joint-acc; regression snapshot at
# best val mean-R2. Avoids penalizing the regression readout with a classification-chosen epoch.
best_valj, best_cls = -1.0, None
best_valr, best_reg = -1e9, None
for ep in range(EPOCHS):
    net.train()
    for x, r, c, y in tl:
        x, r, c, y = x.to(DEV), r.to(DEV), c.to(DEV), y.to(DEV)
        opt.zero_grad()
        pr, pc, pg = net(x)
        (ce(pr, r) + ce(pc, c) + REG_WEIGHT * mse(pg, y)).backward()
        opt.step()
    sched.step()
    ve, te = evaluate(vl), evaluate(tel)
    if ve["joint"] > best_valj:
        best_valj, best_cls = ve["joint"], te
    vr = (ve["atc_r2"] + ve["iptg_r2"]) / 2.0
    if vr > best_valr:
        best_valr, best_reg = vr, te
print(f"[done] backbone={BACKBONE} mode={MODE_TAG} K={K} seed={SEED} n_train={len(train_items)} | "
      f"cls row/col/joint={best_cls['row']:.3f}/{best_cls['col']:.3f}/{best_cls['joint']:.3f} | "
      f"reg MAE aTc/IPTG={best_reg['atc_mae']:.3f}/{best_reg['iptg_mae']:.3f} R2={best_reg['atc_r2']:.3f}/{best_reg['iptg_r2']:.3f}", flush=True)

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
new = not os.path.exists(RESULTS)
with open(RESULTS, "a", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    if new:
        w.writerow(["mode", "K", "seed", "n_train", "row_acc", "col_acc", "joint_acc",
                    "atc_mae", "iptg_mae", "atc_r2", "iptg_r2", "best_val_joint", "best_val_r2"])
    w.writerow([MODE_TAG, K, SEED, len(train_items), f"{best_cls['row']:.4f}", f"{best_cls['col']:.4f}", f"{best_cls['joint']:.4f}",
                f"{best_reg['atc_mae']:.4f}", f"{best_reg['iptg_mae']:.4f}", f"{best_reg['atc_r2']:.4f}", f"{best_reg['iptg_r2']:.4f}",
                f"{best_valj:.4f}", f"{best_valr:.4f}"])
