"""Downstream UTILITY test: predict quantitative COLONY FEATURES from a branching-colony image.
Targets (rotation/scale-sensitive morphology, from colony_features.py): shape_factor, fractal_dim,
log-lacunarity, solidity, eccentricity, edge_std. This is the regime a generative augmenter should
beat plain rotation, because rotation copies carry the SAME (rotation-invariant) label -> no label
diversity, while each synth replicate carries its OWN computed features -> genuine (image,label) diversity.

Scarcity axis = number of TRAIN conditions (branching has ~2-3 reps/condition). Fixed held-out VAL and
TEST conditions (no leakage; the augmenter never trained on these 28). Compares:
  MODE=real       -> real images of N train conditions
  MODE=classical  -> real + N_PER rotated/flipped copies per real (same feature label)
  MODE=augmenter  -> real + N_PER augmenter synth per real (each with its own computed label)
Model selection on VAL mean-R2; report TEST R2 per feature + mean. Appends one row to RESULTS_TSV.

Env: N_TRAIN (5|10|all), MODE, SEED, N_PER[8], EPOCHS[60], BACKBONE[resnet18|resnet50],
     FEATURES_CSV, SYNTH_DIR, REAL_SRC_DIR, REAL_DIR, RESULTS_TSV. DCC GPU."""
import os, glob, re, csv, random
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

TRAIN_REAL = os.environ.get("TRAIN_REAL_DIR", "/hpc/group/youlab/sa603/data/branching_256/canon")
TEST_SRC = os.environ.get("TEST_SRC_DIR", "/hpc/group/youlab/sa603/data/branching_256/test/sources")
TEST_REAL = os.environ.get("TEST_REAL_DIR", "/hpc/group/youlab/sa603/data/branching_256/test/reals")
SYNTH = os.environ.get("SYNTH_DIR", "/hpc/group/youlab/sa603/data/branching_downstream/synth_train_generalist")
FEATURES_CSV = os.environ["FEATURES_CSV"]  # name -> features (reals + synth), from colony_features.py
RESULTS = os.environ.get("RESULTS_TSV", "/hpc/group/youlab/sa603/code/Data_augmentation/branching_downstream_out/results.tsv")
N_TRAIN = os.environ.get("N_TRAIN", "all")
MODE = os.environ.get("MODE", "real")
MODE_TAG = os.environ.get("MODE_TAG", MODE)
SEED = int(os.environ.get("SEED", "0"))
N_PER = int(os.environ.get("N_PER", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "60"))
BACKBONE = os.environ.get("BACKBONE", "resnet18")
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
FEATS = ["shape_factor", "fractal_dim", "solidity", "eccentricity", "edge_std"]  # dropped lacunarity: heavy-tailed, destabilizes R2
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ---- feature labels (rotation-invariant, so a rotated real keeps its label) ----
def _vec(row):
    lac = float(row["lacunarity"])
    d = dict(row); d["lac_log"] = np.log1p(lac)
    return np.array([float(d[f]) for f in FEATS], np.float32)
labels = {}
for row in csv.DictReader(open(FEATURES_CSV)):
    labels[row["name"]] = _vec(row)
print(f"[labels] {len(labels)} images labeled with {FEATS}", flush=True)

def load_rgb(fp):
    return cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
def rot_flip(img, ang, flip):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return cv2.flip(out, 1) if flip else out

# ---- test conditions (held out from the augmenter: the 28) ----
test_by = {}
for fp in glob.glob(TEST_SRC + "/*_src.TIF") + glob.glob(TEST_REAL + "/*_Rep*.TIF"):
    b = os.path.basename(fp); c = b.split("_")[0]
    if b in labels: test_by.setdefault(c, []).append((fp, b))
test_conds = set(test_by)
# ---- train pool = canon conditions NOT in test (disjoint => no leakage) ----
pool_by = {}
for fp in glob.glob(TRAIN_REAL + "/*_Rep*.TIF") + glob.glob(TRAIN_REAL + "/*_src.TIF"):
    b = os.path.basename(fp); c = b.split("_")[0]
    if c in test_conds or b not in labels: continue
    pool_by.setdefault(c, []).append((fp, b))
pool_conds = sorted(pool_by, key=lambda c: int(re.sub(r"\D", "", c) or 0))
VAL = [c for i, c in enumerate(pool_conds) if i % 7 == 0]          # fixed ~1/7 for model selection
trainpool = [c for c in pool_conds if c not in set(VAL)]
random.Random(SEED).shuffle(trainpool)
n_tr = len(trainpool) if N_TRAIN == "all" else min(int(N_TRAIN), len(trainpool))
TRAIN = trainpool[:n_tr]
print(f"[split] pool {len(pool_conds)} conds -> train {len(TRAIN)} val {len(VAL)} | test {len(test_conds)} (held-out)", flush=True)

def real_items(by, cond_list):
    return [(load_rgb(fp), labels[b]) for c in cond_list for fp, b in by[c]]

train_items = []
for c in TRAIN:
    for fp, b in pool_by[c]:
        img = load_rgb(fp); train_items.append((img, labels[b]))
        if MODE == "classical":                    # rotated copies keep the SAME (rot-invariant) label
            rng = random.Random((hash(b) ^ SEED) & 0xffffffff)
            for _ in range(N_PER):
                train_items.append((rot_flip(img, rng.uniform(0, 360), rng.random() < 0.5), labels[b]))
        elif MODE == "augmenter":                  # each synth carries its OWN computed label
            tag = b.split("_", 1)[1].replace(".TIF", "") if "_" in b else "src"
            for s in sorted(glob.glob(f"{SYNTH}/{c}_{tag}_s*.png"))[:N_PER]:
                sb = os.path.basename(s)
                if sb in labels: train_items.append((load_rgb(s), labels[sb]))
val_items = real_items(pool_by, VAL)
test_items = real_items(test_by, sorted(test_conds))
print(f"[data] train {len(train_items)} (mode={MODE_TAG}) val {len(val_items)} test {len(test_items)}", flush=True)

# ---- standardize targets on TRAIN (R2 is scale-invariant; keeps MSE balanced across features) ----
TY = np.stack([y for _, y in train_items])
mu, sd = TY.mean(0), TY.std(0) + 1e-6
def z(y): return (y - mu) / sd

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORM])
EVAL_TF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                              transforms.ToTensor(), NORM])

class DS(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        img, y = self.items[i]; return self.tf(img), torch.tensor(z(y), dtype=torch.float32)

NW = int(os.environ.get("NW", "4"))  # set NW=0 for macOS/local (spawn re-imports the module)
tl = DataLoader(DS(train_items, TRAIN_TF), batch_size=32, shuffle=True, num_workers=NW, drop_last=len(train_items) > 32)
vl = DataLoader(DS(val_items, EVAL_TF), batch_size=64, num_workers=NW)
tel = DataLoader(DS(test_items, EVAL_TF), batch_size=64, num_workers=NW)

class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        if BACKBONE == "resnet50":
            ctor, weights, feat = resnet50, ResNet50_Weights.IMAGENET1K_V2, 2048
        else:
            ctor, weights, feat = resnet18, ResNet18_Weights.IMAGENET1K_V1, 512
        try: self.b = ctor(weights=weights)
        except Exception as e: print("[warn] no pretrained weights:", e, flush=True); self.b = ctor(weights=None)
        self.b.fc = nn.Identity(); self.head = nn.Linear(feat, len(FEATS))
    def forward(self, x): return self.head(self.b(x))

net = Reg().to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
mse = nn.MSELoss()

@torch.no_grad()
def evaluate(loader):
    net.eval(); P, T = [], []
    for x, y in loader:
        P.append(net(x.to(DEV)).cpu().numpy()); T.append(y.numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    ss_res = ((P - T) ** 2).sum(0); ss_tot = ((T - T.mean(0)) ** 2).sum(0) + 1e-9
    r2 = 1 - ss_res / ss_tot                       # per-feature R2 (standardized == raw scale)
    return r2, float(r2.mean())

best_val, best_test, best_vecs = -1e9, None, None
for ep in range(EPOCHS):
    net.train()
    for x, y in tl:
        opt.zero_grad(); mse(net(x.to(DEV)), y.to(DEV)).backward(); opt.step()
    sched.step()
    _, vr = evaluate(vl); tr2, tm = evaluate(tel)
    if vr > best_val:
        best_val, best_test, best_vecs = vr, tm, tr2
print(f"[done] backbone={BACKBONE} mode={MODE_TAG} N_train={N_TRAIN} seed={SEED} n={len(train_items)} | "
      f"TEST meanR2={best_test:.3f} per-feat=" + ",".join(f"{f}:{v:.2f}" for f, v in zip(FEATS, best_vecs)), flush=True)

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
new = not os.path.exists(RESULTS)
with open(RESULTS, "a", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    if new: w.writerow(["mode", "N_train", "seed", "n_train_imgs", "mean_r2", *[f"r2_{f}" for f in FEATS], "best_val_r2"])
    w.writerow([MODE_TAG, N_TRAIN, SEED, len(train_items), f"{best_test:.4f}", *[f"{v:.4f}" for v in best_vecs], f"{best_val:.4f}"])
